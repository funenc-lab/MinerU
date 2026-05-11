# Copyright (c) Opendatalab. All rights reserved.

import base64
import html
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib import error, request

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from mineru.utils.enum_class import BlockType, SplitFlag


TEXT_LEAF_TOP_TYPES = {
    BlockType.TEXT,
    BlockType.LIST,
    BlockType.TITLE,
    BlockType.REF_TEXT,
    BlockType.PHONETIC,
}
TEXT_LEAF_TYPES = {
    BlockType.TEXT,
    BlockType.LIST,
    BlockType.TITLE,
    BlockType.REF_TEXT,
    BlockType.PHONETIC,
}
SKIP_TOP_TYPES = {
    BlockType.TABLE,
    BlockType.IMAGE,
    BlockType.CHART,
    BlockType.INTERLINE_EQUATION,
    BlockType.CODE,
}
PROMPT_VERSION = "vlm_layout_text_adjudicate_v2"
DEFAULT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
RENDER_SCALE = 2.0
PAGE_IMAGE_KEY = "_cross_page_text_vlm_page_image"
REQUEST_RETRY_DELAYS_SECONDS = tuple(range(1, 9))
DEFAULT_REQUEST_CONCURRENCY = 3
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "resources" / "templates"
REPORT_TEMPLATE_PATH = TEMPLATE_DIR / "cross_page_text_vlm_report.html"
REPORT_CARD_TEMPLATE_PATH = TEMPLATE_DIR / "cross_page_text_vlm_report_card.html"


PROMPT_TEMPLATE = """You are a page-break MERGE judge for document reconstruction.
Return exactly one JSON object matching the schema. No markdown and no text outside JSON.

Data source note:
- The candidate text and boxes come from MinerU middle_json pdf_info[*].preproc_blocks.
- This is the original page-level block stream after OCR/image replacement and before para_blocks text/table merging.
- Do not infer from markdown or any already merged/deleted output.

This system uses a three-stage VLM chain:
1. layout_filter: judge body-X alignment and whether B starts a new paragraph/list/clause/heading/label block.
2. text_continuity: judge whether A and B read as continuous text in the same body flow.
3. adjudication: combine stage results into the final merge JSON.

The final adjudication decides whether B should be appended directly to A as the same text block.

Output fields:
- is_continuation: boolean
- confidence: number from 0 to 1
- target: "previous_leaf" if true, otherwise "none"
- a_last_line: string
- b_first_line: string
- joined_preview: string, maximum 240 characters, no newline characters
- b_starts_new_unit: boolean, true when B starts a separate text block/document unit relative to A
- same_list_item_body: boolean, true when the local repeated-item structure shows B continues the same item body
- decision_basis: short reason consistent with is_continuation, maximum 120 characters

Metadata:
{metadata}
"""

LAYOUT_FILTER_PROMPT_TEMPLATE = """You are stage 1 of cross-page text merging: the layout and identifier filter.
Return exactly one JSON object matching the schema. Do not output any other text.

Task: judge only whether B looks like the start of a new text block. Do not make the final merge decision.

Check:
- Whether A's continued body-text start X and B's first body-text start X are approximately aligned. Compare the prose body-text start X, not the identifier, bullet, clause number, or label X.
- Whether A is only a heading, label, or container name while B is its body text.
- Whether B starts a new paragraph.
- Whether B introduces a new list number, clause number, bullet, heading, or label.
- If B has a new identifier, classify it as sibling, child, parent, heading, label, or unclear.

Hard constraints:
- If B has a new list number, clause number, bullet, heading, or label, set b_has_new_list_or_clause_marker=true.
- If new_marker_type is sibling, child, parent, heading, or label, set b_starts_new_block_by_layout=true, unless the identifier itself is visibly split by the page break.
- If A is a heading, label, or container name and B is its body text, set new_marker_type=\"heading\" or \"label\" and b_starts_new_block_by_layout=true.
- Approximate body-X alignment cannot override a new explicit identifier, heading, or label.
- For numbered/list items, ignore the marker column when estimating body-X alignment. B can be aligned with A's prose body even when A's full bbox starts farther left because it includes the marker.
- A page-top paragraph with no new identifier is not a new block merely because its X is shifted from A's full bbox; first compare B with A's prose body-text start X.

Notes:
- Page top position, capitalization, proper nouns, technical terms, sentence-like starts, and topic shifts are not sufficient new-block evidence.
- Inline comma/semicolon enumerated objects inside the same sentence or list body are not separate document blocks.
- Use only red-box text, Metadata OCR, and bboxes.

Output fields:
- body_x_alignment: "aligned" | "shifted" | "unclear"
- b_has_new_paragraph_start: boolean
- b_has_new_list_or_clause_marker: boolean
- new_marker_type: "none" | "sibling" | "child" | "parent" | "heading" | "label" | "unclear"
- b_starts_new_block_by_layout: boolean
- evidence: short reason

Metadata:
{metadata}
"""

TEXT_CONTINUITY_PROMPT_TEMPLATE = """You are stage 2 of cross-page text merging: the text-continuity judge.
Return exactly one JSON object matching the schema. Do not output any other text.

Task: judge only whether the body text in A and B is continuous. Do not make the final structural veto for new lists or clauses.

Check:
- Whether A is textually unclosed, such as ending with a comma, semicolon, connector, preposition, open phrase, or unfinished list body.
- Whether B supplies A's missing object, complement, modifier, following body text, or same-item body.
- Whether A and B look like body text governed by the same explicit identifier.
- Whether A and B are consecutive body paragraphs/sentences under the same numbered/list item.
- Whether B is only an inline enumeration continuation inside the same sentence or list body.
- Whether A and B form a split proper name, acronym, domain-specific term, location, or noun phrase.
- Whether A ends with a verb, preposition, or connector that needs an object/complement and B starts with the noun phrase that supplies it.

Hard constraints:
- A page break may split the body of one numbered/list item into multiple paragraphs or sentences. If A belongs to a numbered/list item and B has no explicit new list/clause/heading/label marker, treat B as same_body_flow=true when B reads as additional body text of the same item, even if A ends with a complete sentence and B starts a new sentence.
- A clause/list number at the beginning of A does not by itself make A a heading or container. If the marker is followed by prose body text, treat A as a numbered item body unless visual layout clearly shows it is only a standalone heading or label.
- If A is a heading, label, or container name and B is its body text, the relation is structural, not body-text continuity; set same_body_flow=false.
- If A is only a list lead-in and B is the first child item after it, the relation is hierarchical, not same_body_flow; set same_body_flow=false.
- If B introduces a new child, sibling, or parent identifier, do not set same_body_flow=true merely because the content is semantically related or expands the topic.
- Only inline comma/semicolon enumerated objects may be inline_enumeration. Child items with explicit list/clause markers are not inline_enumeration.
- If A is textually unclosed and B has no explicit new list/clause/heading/label marker, default to same_body_flow=true unless A is a heading/label/container or B is clearly a child item.
- If A belongs to a numbered/list item and B has no explicit new marker, do not reject same_body_flow merely because A ends with a period, B starts with a capital letter, or B appears at the top of the next page.
- If A ends with a comma or semicolon, B has no explicit new identifier, and B names the next object in the same sentence enumeration, set same_body_flow=true and continuation_type=\"inline_enumeration\".
- If A ends with an incomplete term or noun phrase and B completes that term or noun phrase, set same_body_flow=true.
- If A ends with a verb or prepositional structure that needs an object and B starts with a noun phrase that supplies it, set same_body_flow=true.
- If A ends with an acronym, proper noun, or domain-specific term and B starts with an entity, role, location, component, qualifier, or other noun-phrase continuation, and together they can form one domain-specific noun phrase, set same_body_flow=true.
- If A ends with a comma and B starts with a noun phrase, and B has no explicit list/clause/heading/label marker, prefer treating B as the same sentence enumeration instead of a new sentence.
- If A ends with an open predicate or verb phrase and B starts with a noun phrase, acronym, domain-specific term, or proper noun, first judge whether B supplies A's object/content, even if B later contains a finite verb.
- Do not reject continuity merely because B itself looks like a complete sentence, contains a finite verb, introduces a new proper noun, or looks like a new paragraph.

Notes:
- Do not reject continuity merely because B is on a new page, looks like a new paragraph, starts with capitalization, contains proper nouns/technical terms, or shifts topic.
- Use only red-box text, Metadata OCR, and bboxes.

Output fields:
- a_is_textually_unclosed: boolean
- b_completes_a: boolean
- same_body_flow: boolean
- continuation_type: "same_identifier_body" | "unfinished_phrase" | "inline_enumeration" | "same_item_body" | "none" | "unclear"
- evidence: short reason

Metadata:
{metadata}
"""

ADJUDICATION_PROMPT_TEMPLATE = """You are stage 3 of cross-page text merging: the final adjudicator.
Return exactly one JSON object matching the schema. Do not output any other text.

Task: use the layout/identifier filter result and the text-continuity result to decide whether B should be appended directly to A.

Adjudication rules:
1. If layout_filter.new_marker_type is sibling, child, parent, heading, or label, do not merge unless the identifier itself is visibly split by the page break.
2. If layout_filter.b_has_new_list_or_clause_marker=true and new_marker_type is not none, do not merge.
3. If layout_filter.new_marker_type=\"none\" and text_continuity.a_is_textually_unclosed=true, merge. Page top position, new paragraph appearance, body-X shift, capitalization, new proper nouns, and complete-sentence appearance cannot veto the merge.
4. If layout_filter.new_marker_type=\"none\" and text_continuity.same_body_flow=true, merge.
5. If B has no explicit new marker and appears to continue the body of A's same numbered/list item, merge even if text_continuity.a_is_textually_unclosed=false.
6. If text_continuity shows unfinished_phrase, inline_enumeration, same_identifier_body, or same_item_body, and B has no new list/clause/heading/label marker, merge.
7. If text_continuity.a_is_textually_unclosed=false and text_continuity.same_body_flow=false, and layout_filter.b_starts_new_block_by_layout=true, do not merge.
8. If layout_filter only finds page top position, new paragraph appearance, capitalization, sentence-like start, proper nouns, technical terms, topic shift, or body-X shift, that is not sufficient new-block evidence.
9. If the two stages conflict, prefer explicit identifiers. Without explicit identifiers, prefer textual unclosedness, same-item body continuation, and text continuity.

Required checks:
- Copy A's last line into a_last_line.
- Copy B's first line into b_first_line.
- joined_preview must be a newline-free local A+B preview.
- For merge_false, decision_basis must cite positive new-block evidence.
- All output fields must be mutually consistent.

Output fields:
- is_continuation: boolean
- confidence: number from 0 to 1
- target: "previous_leaf" when true, otherwise "none"
- a_last_line: string
- b_first_line: string
- joined_preview: string, maximum 240 characters, no newline characters
- b_starts_new_unit: boolean
- same_list_item_body: boolean
- decision_basis: short reason

Layout filter result:
{layout_decision}

Text continuity result:
{text_decision}

Metadata:
{metadata}
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hybrid_cross_page_text_continuation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "is_continuation": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "target": {"type": "string", "enum": ["previous_leaf", "none"]},
                "a_last_line": {"type": "string"},
                "b_first_line": {"type": "string"},
                "joined_preview": {"type": "string", "maxLength": 240},
                "b_starts_new_unit": {"type": "boolean"},
                "same_list_item_body": {"type": "boolean"},
                "decision_basis": {"type": "string", "maxLength": 240},
            },
            "required": [
                "is_continuation",
                "confidence",
                "target",
                "a_last_line",
                "b_first_line",
                "joined_preview",
                "b_starts_new_unit",
                "same_list_item_body",
                "decision_basis",
            ],
        },
    },
}

LAYOUT_FILTER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hybrid_cross_page_layout_filter",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "body_x_alignment": {"type": "string", "enum": ["aligned", "shifted", "unclear"]},
                "b_has_new_paragraph_start": {"type": "boolean"},
                "b_has_new_list_or_clause_marker": {"type": "boolean"},
                "new_marker_type": {
                    "type": "string",
                    "enum": ["none", "sibling", "child", "parent", "heading", "label", "unclear"],
                },
                "b_starts_new_block_by_layout": {"type": "boolean"},
                "evidence": {"type": "string", "maxLength": 240},
            },
            "required": [
                "body_x_alignment",
                "b_has_new_paragraph_start",
                "b_has_new_list_or_clause_marker",
                "new_marker_type",
                "b_starts_new_block_by_layout",
                "evidence",
            ],
        },
    },
}

TEXT_CONTINUITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hybrid_cross_page_text_continuity",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "a_is_textually_unclosed": {"type": "boolean"},
                "b_completes_a": {"type": "boolean"},
                "same_body_flow": {"type": "boolean"},
                "continuation_type": {
                    "type": "string",
                    "enum": [
                        "same_identifier_body",
                        "unfinished_phrase",
                        "inline_enumeration",
                        "same_item_body",
                        "none",
                        "unclear",
                    ],
                },
                "evidence": {"type": "string", "maxLength": 240},
            },
            "required": [
                "a_is_textually_unclosed",
                "b_completes_a",
                "same_body_flow",
                "continuation_type",
                "evidence",
            ],
        },
    },
}


def is_enabled():
    value = os.getenv("MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_ENABLE", "false")
    return value.lower() in {"true", "1", "yes"}


def apply_vlm_cross_page_text_merge(pdf_info_list, page_images=None, judge=None, report_dir=None):
    reporter = _VlmReportWriter(report_dir) if report_dir else None
    candidates = list(iter_cross_page_text_candidates(
        pdf_info_list,
        page_images=page_images,
        include_skipped=reporter is not None,
    ))
    if not candidates:
        if reporter is not None:
            reporter.finish()
        return []

    concurrency = _request_concurrency()
    judgeable_count = sum(1 for candidate in candidates if not candidate.get("skip_reason"))
    skipped_count = len(candidates) - judgeable_count
    logger.info(
        "VLM cross-page text merge begin: total_boundaries={}, judgeable={}, skipped={}, concurrency={}",
        len(candidates),
        judgeable_count,
        skipped_count,
        concurrency,
    )

    results = [None] * len(candidates)
    judge_items = []
    for index, candidate in enumerate(candidates):
        if candidate.get("skip_reason"):
            record = {
                "pair": candidate["pair"],
                "prev_page_idx": candidate["prev_page_idx"],
                "current_page_idx": candidate["current_page_idx"],
                "decision": None,
                "merge_status": "merge_unknown",
                "skip_reason": candidate["skip_reason"],
                "error": candidate["skip_reason"],
            }
            logger.info(
                "VLM cross-page text judge skipped: pair={}, reason={}",
                candidate["pair"],
                candidate["skip_reason"],
            )
            results[index] = (record, _prompt_for_skipped_boundary(candidate))
            continue
        prompt = _prompt_for_candidate(candidate)
        judge_items.append((index, candidate, prompt))

    if judge_items and concurrency > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, len(judge_items))) as executor:
            future_to_index = {
                executor.submit(_judge_candidate, candidate, prompt, judge): index
                for index, candidate, prompt in judge_items
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
    else:
        for index, candidate, prompt in judge_items:
            results[index] = _judge_candidate(candidate, prompt, judge)

    decisions = []
    merge_redirects = {}
    for index, candidate in enumerate(candidates):
        record, prompt = results[index]
        if reporter is not None:
            reporter.record(candidate, record, prompt)
        decisions.append(record)
        if record.get("decision") and record["decision"].get("is_continuation") is True:
            previous_leaf = _resolve_redirected_leaf(candidate["previous_leaf"], merge_redirects)
            current_leaf = _resolve_redirected_leaf(candidate["current_leaf"], merge_redirects)
            if current_leaf is previous_leaf:
                continue
            _merge_current_leaf_into_previous(current_leaf, previous_leaf)
            merge_redirects[id(candidate["current_leaf"])] = previous_leaf
            merge_redirects[id(current_leaf)] = previous_leaf
    if reporter is not None:
        reporter.finish()
    logger.info(
        "VLM cross-page text merge finished: total={}, merge_true={}, merge_false={}, merge_unknown={}",
        len(decisions),
        sum(1 for record in decisions if record.get("merge_status") == "merge_true"),
        sum(1 for record in decisions if record.get("merge_status") == "merge_false"),
        sum(1 for record in decisions if record.get("merge_status") == "merge_unknown"),
    )
    return decisions


def _judge_candidate(candidate, prompt, judge):
    started_at = time.time()
    logger.info(
        "VLM cross-page text judge start: pair={}, model={}, api_url={}",
        candidate["pair"],
        _model_name(),
        _api_url(),
    )
    try:
        decision = judge(candidate) if judge is not None else _judge_with_lmstudio(candidate, prompt=prompt)
    except Exception as exc:
        elapsed = round(time.time() - started_at, 2)
        logger.warning(
            "VLM cross-page text judge failed: pair={}, elapsed={}s, error={}: {}",
            candidate["pair"],
            elapsed,
            type(exc).__name__,
            exc,
        )
        record = {
            "pair": candidate["pair"],
            "prev_page_idx": candidate["prev_page_idx"],
            "current_page_idx": candidate["current_page_idx"],
            "decision": None,
            "merge_status": "merge_unknown",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return record, prompt

    merge_status = "merge_true" if decision and decision.get("is_continuation") is True else "merge_false"
    elapsed = round(time.time() - started_at, 2)
    logger.info(
        "VLM cross-page text judge done: pair={}, status={}, confidence={}, elapsed={}s",
        candidate["pair"],
        merge_status,
        (decision or {}).get("confidence"),
        elapsed,
    )
    record = {
        "pair": candidate["pair"],
        "prev_page_idx": candidate["prev_page_idx"],
        "current_page_idx": candidate["current_page_idx"],
        "decision": decision,
        "merge_status": merge_status,
    }
    return record, prompt


def _resolve_redirected_leaf(leaf, redirects):
    current = leaf
    seen = set()
    while id(current) in redirects and id(current) not in seen:
        seen.add(id(current))
        current = redirects[id(current)]
    return current


def _request_concurrency():
    value = os.getenv("MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_CONCURRENCY")
    if value is None:
        return DEFAULT_REQUEST_CONCURRENCY
    try:
        concurrency = int(value)
    except ValueError:
        logger.warning(
            "Invalid MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_CONCURRENCY value: {}, using default {}",
            value,
            DEFAULT_REQUEST_CONCURRENCY,
        )
        return DEFAULT_REQUEST_CONCURRENCY
    if concurrency < 1:
        logger.warning(
            "Invalid MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_CONCURRENCY value: {}, using 1",
            value,
        )
        return 1
    return concurrency


def iter_cross_page_text_candidates(pdf_info_list, page_images=None, include_skipped=False):
    leaves_by_page = [list(_walk_page_text_leaves(page_info)) for page_info in pdf_info_list]
    for prev_idx in range(len(pdf_info_list) - 1):
        cur_idx = prev_idx + 1
        previous_leaves = leaves_by_page[prev_idx]
        current_leaves = leaves_by_page[cur_idx]
        if not previous_leaves or not current_leaves:
            if include_skipped:
                yield _make_skipped_boundary(
                    pdf_info_list,
                    page_images,
                    prev_idx,
                    cur_idx,
                    "no_text_leaf_on_previous_page" if not previous_leaves else "no_text_leaf_on_current_page",
                )
            continue

        previous_source_leaf = previous_leaves[-1]
        current_source_leaf = current_leaves[0]
        previous_leaf = _resolve_merge_leaf(pdf_info_list[prev_idx], previous_source_leaf)
        current_leaf = _resolve_merge_leaf(pdf_info_list[cur_idx], current_source_leaf)
        if previous_leaf is None or current_leaf is None:
            if include_skipped:
                yield _make_skipped_boundary(
                    pdf_info_list,
                    page_images,
                    prev_idx,
                    cur_idx,
                    "source_leaf_not_mapped_to_para_blocks",
                    previous_source_leaf=previous_source_leaf,
                    current_source_leaf=current_source_leaf,
                )
            continue
        candidate = {
            "pair": f"{prev_idx:03d}-{cur_idx:03d}",
            "prev_page_idx": prev_idx,
            "current_page_idx": cur_idx,
            "previous_page_size": pdf_info_list[prev_idx].get("page_size"),
            "current_page_size": pdf_info_list[cur_idx].get("page_size"),
            "previous_leaf": previous_leaf,
            "current_leaf": current_leaf,
            "previous_source_leaf": previous_source_leaf,
            "current_source_leaf": current_source_leaf,
            "metadata": _make_metadata(prev_idx, cur_idx, previous_source_leaf, current_source_leaf),
        }
        if page_images:
            candidate["image"] = _create_candidate_image(candidate, page_images)
        yield candidate


def _make_skipped_boundary(
    pdf_info_list,
    page_images,
    prev_idx,
    cur_idx,
    skip_reason,
    previous_source_leaf=None,
    current_source_leaf=None,
):
    candidate = {
        "pair": f"{prev_idx:03d}-{cur_idx:03d}",
        "prev_page_idx": prev_idx,
        "current_page_idx": cur_idx,
        "previous_page_size": pdf_info_list[prev_idx].get("page_size"),
        "current_page_size": pdf_info_list[cur_idx].get("page_size"),
        "previous_source_leaf": previous_source_leaf,
        "current_source_leaf": current_source_leaf,
        "skip_reason": skip_reason,
        "metadata": {
            "prompt_version": PROMPT_VERSION,
            "source": "middle_json.pdf_info[*].preproc_blocks text leaves",
            "prev_page_idx": prev_idx,
            "current_page_idx": cur_idx,
            "skip_reason": skip_reason,
        },
    }
    if page_images:
        candidate["image"] = _create_boundary_context_image(candidate, page_images)
    return candidate


def _walk_page_text_leaves(page_info):
    blocks = page_info.get("preproc_blocks") or page_info.get("para_blocks", [])
    for block_index, block in enumerate(blocks):
        yield from _walk_text_leaves(block, block, [block_index])


def _walk_text_leaves(block, top_block, path):
    top_type = top_block.get("type")
    if top_type in SKIP_TOP_TYPES or top_type not in TEXT_LEAF_TOP_TYPES:
        return

    if _is_text_leaf(block):
        yield {
            "top_type": top_type,
            "leaf_type": block.get("type"),
            "top_index": top_block.get("index"),
            "path": path,
            "block": block,
            "bbox": _line_bbox_union(block) or block.get("bbox"),
            "lines": _line_texts(block),
        }

    for child_index, child in enumerate(block.get("blocks", [])):
        yield from _walk_text_leaves(child, top_block, [*path, child_index])


def _is_text_leaf(block):
    if block.get("type") not in TEXT_LEAF_TYPES:
        return False
    return _block_has_lines(block)


def _block_has_lines(block):
    return any(line.get("spans") for line in block.get("lines", []))


def _line_bbox_union(block):
    bboxes = [line.get("bbox") for line in block.get("lines", []) if line.get("bbox")]
    if not bboxes:
        return None
    return [
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    ]


def _line_texts(block):
    lines = []
    for line in block.get("lines", []):
        text = "".join(str(span.get("content", "")) for span in line.get("spans", []))
        text = text.strip()
        if text:
            lines.append(text)
    return lines


def _resolve_merge_leaf(page_info, source_leaf):
    para_blocks = page_info.get("para_blocks")
    if not para_blocks:
        return source_leaf["block"]
    block = _block_at_path(para_blocks, source_leaf["path"])
    if block is None or not _is_text_leaf(block):
        return None
    return block


def _block_at_path(blocks, path):
    current_blocks = blocks
    current_block = None
    for index in path:
        if index >= len(current_blocks):
            return None
        current_block = current_blocks[index]
        current_blocks = current_block.get("blocks", [])
    return current_block


def _make_metadata(prev_idx, cur_idx, previous_leaf, current_leaf):
    previous_lines = previous_leaf["lines"]
    current_lines = current_leaf["lines"]
    return {
        "prompt_version": PROMPT_VERSION,
        "source": "middle_json.pdf_info[*].preproc_blocks text leaves",
        "prev_page_idx": prev_idx,
        "current_page_idx": cur_idx,
        "previous_top_type": previous_leaf.get("top_type"),
        "previous_leaf_type": previous_leaf.get("leaf_type"),
        "previous_leaf_path": previous_leaf.get("path"),
        "current_top_type": current_leaf.get("top_type"),
        "current_leaf_type": current_leaf.get("leaf_type"),
        "current_leaf_path": current_leaf.get("path"),
        "region_A_previous_leaf": {
            "bbox": previous_leaf.get("bbox"),
            "last_lines": previous_lines[-3:],
            "a_last_line": previous_lines[-1] if previous_lines else "",
        },
        "region_B_current_leaf": {
            "bbox": current_leaf.get("bbox"),
            "first_lines": current_lines[:3],
            "b_first_line": current_lines[0] if current_lines else "",
        },
    }


def _merge_current_leaf_into_previous(current_leaf, previous_leaf):
    moved_lines = current_leaf.get("lines", [])
    for line in moved_lines:
        for span in line.get("spans", []):
            span[SplitFlag.CROSS_PAGE] = True
    previous_leaf.setdefault("lines", []).extend(moved_lines)
    current_leaf["lines"] = []
    current_leaf[SplitFlag.LINES_DELETED] = True


def _prompt_for_candidate(candidate):
    return PROMPT_TEMPLATE.format(metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2))


def _layout_filter_prompt_for_candidate(candidate):
    return LAYOUT_FILTER_PROMPT_TEMPLATE.format(
        metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2)
    )


def _text_continuity_prompt_for_candidate(candidate):
    return TEXT_CONTINUITY_PROMPT_TEMPLATE.format(
        metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2)
    )


def _adjudication_prompt_for_candidate(candidate, layout_decision, text_decision):
    return ADJUDICATION_PROMPT_TEMPLATE.format(
        layout_decision=json.dumps(layout_decision, ensure_ascii=False, indent=2),
        text_decision=json.dumps(text_decision, ensure_ascii=False, indent=2),
        metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2),
    )


def _prompt_for_skipped_boundary(candidate):
    return (
        "Skipped VLM judgment.\n"
        "This page boundary did not produce a direct text-merge candidate; "
        "the report only keeps the context image for manual inspection.\n\n"
        f"Metadata:\n{json.dumps(candidate['metadata'], ensure_ascii=False, indent=2)}\n"
    )


class _VlmReportWriter:
    def __init__(self, report_dir):
        self.report_dir = Path(report_dir)
        self.marked_dir = self.report_dir / "marked"
        self.prompts_dir = self.report_dir / "prompts"
        self.records = []
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.marked_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_template_rel = f"prompt_template_{PROMPT_VERSION}.txt"
        (self.report_dir / self.prompt_template_rel).write_text(PROMPT_TEMPLATE, encoding="utf-8")

    def record(self, candidate, record, prompt):
        pair = record["pair"]
        merge_status = record["merge_status"]

        prompt_rel = Path("prompts") / f"{pair}_prompt.txt"
        (self.report_dir / prompt_rel).write_text(prompt, encoding="utf-8")
        record["prompt_file"] = prompt_rel.as_posix()

        image = candidate.get("image")
        if image is not None:
            image_rel = Path("marked") / f"{pair}_{merge_status}.png"
            image.save(self.report_dir / image_rel)
            record["marked_image"] = image_rel.as_posix()

        report_record = {
            "pair": pair,
            "prev_page_idx": record["prev_page_idx"],
            "current_page_idx": record["current_page_idx"],
            "merge_status": merge_status,
            "decision": record.get("decision"),
            "error": record.get("error"),
            "skip_reason": record.get("skip_reason"),
            "metadata": candidate.get("metadata"),
            "prompt_file": record.get("prompt_file"),
            "marked_image": record.get("marked_image"),
        }
        self.records.append(report_record)

    def finish(self):
        summary = {
            "prompt_version": PROMPT_VERSION,
            "model": _model_name(),
            "api_url": _api_url(),
            "prompt_template": self.prompt_template_rel,
            "total": len(self.records),
            "merge_true": self._count("merge_true"),
            "merge_false": self._count("merge_false"),
            "merge_unknown": self._count("merge_unknown"),
            "skipped_boundary": sum(1 for record in self.records if record.get("skip_reason")),
            "filename_pattern": "page-page_merge-status.png",
            "image_note": "marked images use red boxes for A previous-page tail and B current-page head",
        }
        (self.report_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.report_dir / "results.json").write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.report_dir / "report.html").write_text(
            _render_report_html(summary, self.records, self.report_dir),
            encoding="utf-8",
        )

    def _count(self, merge_status):
        return sum(1 for record in self.records if record.get("merge_status") == merge_status)


def _render_report_html(summary, records, report_dir):
    cards = "\n".join(_render_report_card(record, report_dir) for record in records)
    summary_json = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    replacements = {
        "{{TOTAL}}": html.escape(str(summary["total"])),
        "{{MERGE_TRUE}}": html.escape(str(summary["merge_true"])),
        "{{MERGE_FALSE}}": html.escape(str(summary["merge_false"])),
        "{{MERGE_UNKNOWN}}": html.escape(str(summary["merge_unknown"])),
        "{{MODEL}}": html.escape(str(summary["model"])),
        "{{PROMPT_TEMPLATE}}": html.escape(str(summary["prompt_template"])),
        "{{CARDS}}": cards,
        "{{SUMMARY_JSON}}": summary_json,
    }
    return _replace_template_tokens(_read_template(REPORT_TEMPLATE_PATH), replacements)


def _render_report_card(record, report_dir):
    decision = record.get("decision") or {}
    metadata = record.get("metadata") or {}
    a_line = decision.get("a_last_line") or metadata.get("region_A_previous_leaf", {}).get("a_last_line") or ""
    b_line = decision.get("b_first_line") or metadata.get("region_B_current_leaf", {}).get("b_first_line") or ""
    confidence = decision.get("confidence")
    confidence_text = "N/A" if confidence is None else str(confidence)
    reason = record.get("error") or decision.get("decision_basis") or ""
    skip_reason = record.get("skip_reason")
    if skip_reason:
        reason = f"Skipped VLM judgment: {skip_reason}"
    image_html = ""
    if record.get("marked_image"):
        src = _inline_image_src(report_dir, record["marked_image"]) or record["marked_image"]
        src = html.escape(src)
        image_html = f'<img src="{src}" alt="{html.escape(record["pair"])}">'
    prompt_html = ""
    if record.get("prompt_file"):
        prompt_text = _read_report_text(report_dir, record["prompt_file"])
        prompt_html = (
            f'<details class="prompt"><summary>Prompt: {html.escape(record["prompt_file"])}</summary>'
            f'<pre>{html.escape(prompt_text)}</pre></details>'
        )
    replacements = {
        "{{MERGE_STATUS}}": html.escape(record["merge_status"]),
        "{{PAIR}}": html.escape(record["pair"]),
        "{{STATUS_LABEL}}": _status_label(record["merge_status"]),
        "{{IMAGE_HTML}}": image_html,
        "{{REASON}}": html.escape(str(reason)),
        "{{CONFIDENCE}}": html.escape(confidence_text),
        "{{A_LAST_LINE}}": html.escape(str(a_line)),
        "{{B_FIRST_LINE}}": html.escape(str(b_line)),
        "{{PROMPT_HTML}}": prompt_html,
    }
    return _replace_template_tokens(_read_template(REPORT_CARD_TEMPLATE_PATH), replacements)


def _inline_image_src(report_dir, relative_path):
    path = report_dir / relative_path
    if not path.exists():
        return None
    with Image.open(path) as image:
        with BytesIO() as buffer:
            image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _read_report_text(report_dir, relative_path):
    path = report_dir / relative_path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_template(path):
    return path.read_text(encoding="utf-8")


def _replace_template_tokens(template, replacements):
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def _status_label(merge_status):
    return {
        "merge_true": "Merge",
        "merge_false": "No merge",
        "merge_unknown": "Unknown",
    }.get(merge_status, merge_status)


def _judge_with_lmstudio(candidate, prompt=None):
    image = candidate.get("image")
    if image is None:
        raise RuntimeError("page_images is required when no judge is injected")
    pair = candidate.get("pair")
    layout_prompt = _layout_filter_prompt_for_candidate(candidate)
    layout_decision = _call_model(
        layout_prompt,
        image,
        pair=pair,
        response_schema=LAYOUT_FILTER_SCHEMA,
        stage="layout_filter",
    )
    text_prompt = _text_continuity_prompt_for_candidate(candidate)
    text_decision = _call_model(
        text_prompt,
        image,
        pair=pair,
        response_schema=TEXT_CONTINUITY_SCHEMA,
        stage="text_continuity",
    )
    adjudication_prompt = _adjudication_prompt_for_candidate(
        candidate,
        layout_decision,
        text_decision,
    )
    decision = _call_model(
        adjudication_prompt,
        image,
        pair=pair,
        response_schema=SCHEMA,
        stage="adjudication",
    )
    decision = _normalize_adjudication_from_stage_decisions(candidate, decision, layout_decision, text_decision)
    decision["_stage_decisions"] = {
        "layout_filter": layout_decision,
        "text_continuity": text_decision,
    }
    return decision


def _normalize_adjudication_from_stage_decisions(candidate, decision, layout_decision, text_decision):
    marker_type = layout_decision.get("new_marker_type")
    has_marker = layout_decision.get("b_has_new_list_or_clause_marker") is True
    continuation_type = text_decision.get("continuation_type")
    text_positive = (
        text_decision.get("a_is_textually_unclosed") is True
        or text_decision.get("same_body_flow") is True
        or continuation_type in {
            "same_identifier_body",
            "unfinished_phrase",
            "inline_enumeration",
            "same_item_body",
        }
    )
    text_negative = (
        text_decision.get("a_is_textually_unclosed") is False
        and text_decision.get("same_body_flow") is False
    )
    layout_negative = layout_decision.get("b_starts_new_block_by_layout") is True

    if marker_type in {"sibling", "child", "parent", "heading", "label"} or (has_marker and marker_type != "none"):
        return _with_forced_decision(
            candidate,
            decision,
            is_continuation=False,
            confidence=max(float(decision.get("confidence") or 0), 0.95),
            basis=f"Stage filter found an explicit new-block marker: {layout_decision.get('evidence', '')}"[:120],
        )

    if marker_type == "none" and text_positive:
        if text_decision.get("same_body_flow") is True:
            basis_prefix = "No explicit new marker; text stage found same-block continuation"
        elif text_decision.get("a_is_textually_unclosed") is True:
            basis_prefix = "No explicit new marker; A is unclosed, so progressive filtering merges"
        else:
            basis_prefix = f"No explicit new marker; text continuation type is {continuation_type}"
        return _with_forced_decision(
            candidate,
            decision,
            is_continuation=True,
            confidence=max(float(decision.get("confidence") or 0), 0.9),
            basis=f"{basis_prefix}: {text_decision.get('evidence', '')}"[:120],
        )

    if layout_negative and text_negative:
        return _with_forced_decision(
            candidate,
            decision,
            is_continuation=False,
            confidence=max(float(decision.get("confidence") or 0), 0.9),
            basis=f"Layout and text stages both indicate a new block: {layout_decision.get('evidence', '')}"[:120],
        )

    return decision


def _with_forced_decision(candidate, decision, is_continuation, confidence, basis):
    metadata = candidate.get("metadata") or {}
    a_last_line = (
        decision.get("a_last_line")
        or metadata.get("region_A_previous_leaf", {}).get("a_last_line")
        or ""
    )
    b_first_line = (
        decision.get("b_first_line")
        or metadata.get("region_B_current_leaf", {}).get("b_first_line")
        or ""
    )
    joined_preview = decision.get("joined_preview") or f"{a_last_line} {b_first_line}".strip()
    return {
        **decision,
        "is_continuation": is_continuation,
        "confidence": min(max(confidence, 0), 1),
        "target": "previous_leaf" if is_continuation else "none",
        "a_last_line": a_last_line,
        "b_first_line": b_first_line,
        "joined_preview": joined_preview[:240],
        "b_starts_new_unit": not is_continuation,
        "same_list_item_body": bool(decision.get("same_list_item_body")) if is_continuation else False,
        "decision_basis": basis,
    }


def _call_model(prompt, image, pair=None, response_schema=SCHEMA, stage=None):
    model = _model_name()
    api_url = _api_url()
    api_key = _api_key()
    image_url = "data:image/png;base64," + _encode_image(image)
    payload = _make_chat_payload(model, prompt, image_url, response_schema=response_schema)
    errors = []
    request_start = time.time()
    logger.info(
        "VLM model request begin: pair={}, stage={}, model={}, api_url={}",
        pair,
        stage,
        model,
        api_url,
    )
    for string_image_url in (False, True):
        if string_image_url:
            payload = _make_chat_payload(
                model,
                prompt,
                image_url,
                string_image_url=True,
                response_schema=response_schema,
            )
        for with_schema in (True, False):
            trial_payload = dict(payload)
            if not with_schema:
                trial_payload.pop("response_format", None)
            for attempt_index in range(len(REQUEST_RETRY_DELAYS_SECONDS) + 1):
                try:
                    trial_start = time.time()
                    logger.debug(
                        "VLM model request attempt: pair={}, stage={}, attempt={}, schema={}, string_image_url={}",
                        pair,
                        stage,
                        attempt_index + 1,
                        with_schema,
                        string_image_url,
                    )
                    data = _request_json(api_url, trial_payload, api_key=api_key)
                    message = data["choices"][0]["message"]
                    raw = message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
                    if not raw:
                        logger.info("VLM model request empty content, trying stream: pair={}", pair)
                        raw = _request_stream_text(api_url, trial_payload, api_key=api_key)
                    decision = _parse_model_json(raw)
                    _validate_model_json(decision, response_schema)
                    logger.info(
                        "VLM model request success: pair={}, stage={}, elapsed={}s, attempt_elapsed={}s, schema={}, string_image_url={}",
                        pair,
                        stage,
                        round(time.time() - request_start, 2),
                        round(time.time() - trial_start, 2),
                        with_schema,
                        string_image_url,
                    )
                    return decision
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    if (
                        attempt_index >= len(REQUEST_RETRY_DELAYS_SECONDS)
                        or not _should_retry_request_error(exc)
                    ):
                        logger.warning(
                            "VLM model request attempt failed without retry: pair={}, stage={}, attempt={}, schema={}, string_image_url={}, error={}: {}",
                            pair,
                            stage,
                            attempt_index + 1,
                            with_schema,
                            string_image_url,
                            type(exc).__name__,
                            exc,
                        )
                        break
                    delay = REQUEST_RETRY_DELAYS_SECONDS[attempt_index]
                    logger.warning(
                        "VLM model request retry: pair={}, stage={}, attempt={}, sleep={}s, schema={}, string_image_url={}, error={}: {}",
                        pair,
                        stage,
                        attempt_index + 1,
                        delay,
                        with_schema,
                        string_image_url,
                        type(exc).__name__,
                        exc,
                    )
                    time.sleep(delay)
    raise RuntimeError(" ; ".join(errors))


def _validate_model_json(decision, response_schema):
    schema = (response_schema or {}).get("json_schema", {}).get("schema", {})
    if not schema:
        return
    if not isinstance(decision, dict):
        raise ValueError("model response is not a JSON object")
    properties = schema.get("properties", {})
    for field in schema.get("required", []):
        if field not in decision:
            raise ValueError(f"model response missing required field: {field}")
    for field, value in decision.items():
        field_schema = properties.get(field)
        if field_schema is None:
            if schema.get("additionalProperties") is False:
                raise ValueError(f"model response has unexpected field: {field}")
            continue
        expected_type = field_schema.get("type")
        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"model response field {field} must be boolean")
        if expected_type == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"model response field {field} must be number")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"model response field {field} must be string")
        enum = field_schema.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"model response field {field} has invalid enum value: {value}")


def _should_retry_request_error(exc):
    code = _http_status_code(exc)
    if code is None:
        return True
    return not (400 <= code < 500 and code not in {408, 409, 425, 429})


def _http_status_code(exc):
    if isinstance(exc, error.HTTPError):
        return exc.code
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def _make_chat_payload(model, prompt, image_url, string_image_url=False, response_schema=SCHEMA):
    image_url_value = image_url if string_image_url else {"url": image_url}
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url_value},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 520,
        "response_format": response_schema,
    }


def _model_name():
    return (
        os.getenv("MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_MODEL
    )


def _api_url():
    explicit_url = os.getenv("MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_API_URL")
    if explicit_url:
        return explicit_url

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        return base_url.rstrip("/") + "/chat/completions"

    return DEFAULT_API_URL


def _api_key():
    return (
        os.getenv("MINERU_HYBRID_CROSS_PAGE_TEXT_VLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )


def _request_json(api_url, payload, timeout=180, api_key=""):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _request_stream_text(api_url, payload, timeout=180, api_key=""):
    stream_payload = {**payload, "stream": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(
        api_url,
        data=json.dumps(stream_payload).encode("utf-8"),
        headers=headers,
    )
    chunks = []
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunks.extend(_extract_stream_text_chunks(event))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    return "".join(chunks)


def _extract_stream_text_chunks(event):
    for choice in event.get("choices", []):
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    yield item["text"]

    event_type = event.get("type")
    if event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
        yield event["delta"]
    elif event_type == "response.output_text.done" and isinstance(event.get("text"), str):
        yield event["text"]


def _parse_model_json(raw):
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _encode_image(image):
    with BytesIO() as buffer:
        image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _create_candidate_image(candidate, page_images):
    prev_idx = candidate["prev_page_idx"]
    cur_idx = candidate["current_page_idx"]
    prev_img = _as_pil(page_images[prev_idx])
    cur_img = _as_pil(page_images[cur_idx])
    prev_box = _bbox_to_pixels(candidate["previous_source_leaf"].get("bbox"), candidate["previous_page_size"], prev_img)
    cur_box = _bbox_to_pixels(candidate["current_source_leaf"].get("bbox"), candidate["current_page_size"], cur_img)

    margin = 12
    prev_y0 = max(0, prev_box[1] - margin)
    prev_y1 = min(prev_img.height, prev_box[3] + margin)
    cur_y0 = max(0, cur_box[1] - margin)
    cur_y1 = min(cur_img.height, cur_box[3] + margin)
    prev_crop = prev_img.crop((0, prev_y0, prev_img.width, prev_y1))
    cur_crop = cur_img.crop((0, cur_y0, cur_img.width, cur_y1))

    label_h = 42
    gap = 36
    width = max(prev_crop.width, cur_crop.width)
    height = label_h + prev_crop.height + gap + label_h + cur_crop.height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(22)
    red = (220, 0, 0)
    canvas.paste(prev_crop, (0, label_h))
    cur_offset = label_h + prev_crop.height + gap + label_h
    canvas.paste(cur_crop, (0, cur_offset))
    draw.text((12, 8), f"A previous page tail: page_idx {prev_idx}", fill=red, font=font)
    draw.text((12, label_h + prev_crop.height + gap + 8), f"B current page head: page_idx {cur_idx}", fill=red, font=font)
    _draw_box(draw, prev_box, prev_y0, label_h, "A", font)
    _draw_box(draw, cur_box, cur_y0, cur_offset, "B", font)
    return canvas


def _create_boundary_context_image(candidate, page_images):
    prev_idx = candidate["prev_page_idx"]
    cur_idx = candidate["current_page_idx"]
    prev_img = _as_pil(page_images[prev_idx])
    cur_img = _as_pil(page_images[cur_idx])

    prev_box = _optional_bbox_to_pixels(
        (candidate.get("previous_source_leaf") or {}).get("bbox"),
        candidate.get("previous_page_size"),
        prev_img,
    )
    cur_box = _optional_bbox_to_pixels(
        (candidate.get("current_source_leaf") or {}).get("bbox"),
        candidate.get("current_page_size"),
        cur_img,
    )

    prev_crop, prev_y0 = _edge_crop(prev_img, "bottom", prev_box)
    cur_crop, cur_y0 = _edge_crop(cur_img, "top", cur_box)

    label_h = 42
    gap = 36
    width = max(prev_crop.width, cur_crop.width)
    height = label_h + prev_crop.height + gap + label_h + cur_crop.height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(22)
    red = (220, 0, 0)
    canvas.paste(prev_crop, (0, label_h))
    cur_offset = label_h + prev_crop.height + gap + label_h
    canvas.paste(cur_crop, (0, cur_offset))
    draw.text((12, 8), f"A previous page tail: page_idx {prev_idx}", fill=red, font=font)
    draw.text((12, label_h + prev_crop.height + gap + 8), f"B current page head: page_idx {cur_idx}", fill=red, font=font)
    if prev_box is not None:
        _draw_box(draw, prev_box, prev_y0, label_h, "A", font)
    if cur_box is not None:
        _draw_box(draw, cur_box, cur_y0, cur_offset, "B", font)
    return canvas


def _optional_bbox_to_pixels(bbox, page_size, image):
    if not bbox or not page_size:
        return None
    return _bbox_to_pixels(bbox, page_size, image)


def _edge_crop(image, edge, focus_box=None):
    if focus_box is not None:
        margin = 12
        y0 = max(0, focus_box[1] - margin)
        y1 = min(image.height, focus_box[3] + margin)
        return image.crop((0, y0, image.width, y1)), y0

    crop_height = max(1, int(image.height * 0.30))
    if edge == "bottom":
        y0 = image.height - crop_height
    else:
        y0 = 0
    y1 = min(image.height, y0 + crop_height)
    return image.crop((0, y0, image.width, y1)), y0


def _as_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "img_pil" in image:
        return image["img_pil"].convert("RGB")
    raise TypeError(f"unsupported page image type: {type(image)!r}")


def _bbox_to_pixels(bbox, page_size, image):
    if not bbox or not page_size:
        raise ValueError("bbox and page_size are required for VLM cross-page text image")
    sx = image.width / page_size[0]
    sy = image.height / page_size[1]
    return [int(bbox[0] * sx), int(bbox[1] * sy), int(bbox[2] * sx), int(bbox[3] * sy)]


def _draw_box(draw, box, crop_y0, offset_y, label, font):
    x0, y0, x1, y1 = box
    y0 = y0 - crop_y0 + offset_y
    y1 = y1 - crop_y0 + offset_y
    red = (220, 0, 0)
    for i in range(4):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=red)
    draw.text((max(2, x0 - 28), max(0, y0 - 26)), label, fill=red, font=font)


def _load_font(size):
    for path in (
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()
