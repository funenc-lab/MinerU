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
PROMPT_VERSION = "generic_direct_append_v3"
DEFAULT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
RENDER_SCALE = 2.0
PAGE_IMAGE_KEY = "_cross_page_text_vlm_page_image"
REQUEST_RETRY_DELAYS_SECONDS = tuple(range(1, 9))
DEFAULT_REQUEST_CONCURRENCY = 3


PROMPT_TEMPLATE = """You are a page-break MERGE judge for document reconstruction.
Return exactly one JSON object matching the schema. No markdown and no text outside JSON.

Data source note:
- The candidate text and boxes come from MinerU middle_json pdf_info[*].preproc_blocks.
- This is the original page-level block stream after OCR/image replacement and before para_blocks text/table merging.
- Do not infer from markdown or any already merged/deleted output.

Core definition:
- The task is direct text append, not outline grouping.
- merge_true means B must be appended directly to A with no block or paragraph boundary between them.
- merge_false means A and B should remain separate text blocks, even if they belong to the same topic, section, requirement, or parent structure.

Generic decision principles:
1. A physical page break is not a reading boundary.
2. Weak surface cues are never decisive by themselves: punctuation at the end of A, capitalization at the start of B, proper nouns, technical terms, a new sentence form, a changed topic word, or a line that looks complete.
3. A period or other sentence-ending punctuation is not proof that A is complete. One paragraph, requirement, explanation, or item body may contain multiple sentences across pages.
4. Same topic, same section, parent-child relation, or heading-followed-by-body relation is not enough for merge_true.
5. Return merge_true only when A and B form one continuous text-bearing unit that would be wrong if separated by a block boundary.
6. Return merge_false when A is a standalone label, heading, container name, or structural parent and B is content under it; such content is related but not appended to the label text.
7. For repeated item structures, decide whether B continues the same item body or starts a different item from structure and reading flow. Do not merge different items, and do not reject a same-item continuation merely because the surrounding structure repeats.
8. A sibling item is not direct text append to the previous item. Parent-to-child item transitions are grouping relations, not direct text append. Do not treat adjacent list items as one continuation just because they are consecutive, coordinated, or connected by local wording.
9. When evidence is ambiguous and B has no independent new-unit start, choose merge_true with lower confidence only if direct append would preserve the local text flow better than keeping a boundary.

Prohibited decision basis:
- Do not cite punctuation, uppercase/lowercase, proper nouns, technical terms, page start, or "new sentence" as the deciding reason.
- Do not rely on fixed phrase patterns or hard-coded examples. Judge the actual local visual structure and semantic continuity.
- Do not justify merge_true only by saying B belongs under A's heading or in the same section.
- Do not justify merge_true only by saying B is the next coordinated/listed item after A.
- For merge_false, decision_basis must cite positive local structural evidence that B starts a separate block.
- If the only evidence for merge_false is punctuation, a sentence-looking start, semantic completeness, or a topic shift without a visible structural marker, revise to merge_true when direct append preserves text flow.

Evidence rules:
1. Use only red-box text and Metadata OCR.
2. Do not use red-box outside text.
3. Do not reject continuation merely because B starts a new physical page.
4. The final boolean must match the reason.
5. decision_basis must be written in Chinese, short, and must not include step-by-step hidden reasoning.

Required checks:
- Copy A's last line into a_last_line.
- Copy B's first line into b_first_line.
- Determine whether B should be directly appended to A, not merely grouped under A.
- Determine whether joined_preview is a continuous text-bearing unit without a block boundary.
- If B has no visible structural marker and the only concern is that A looks complete or B starts a new sentence, do not mark B as a new unit.

Output fields:
- is_continuation: boolean
- confidence: number from 0 to 1
- target: "previous_leaf" if true, otherwise "none"
- a_last_line: string
- b_first_line: string
- joined_preview: string, maximum 240 characters, no newline characters
- b_starts_new_unit: boolean, true when B starts a separate text block/document unit relative to A
- same_list_item_body: boolean, true when the local repeated-item structure shows B continues the same item body
- decision_basis: Chinese short reason consistent with is_continuation, maximum 120 Chinese characters

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


def _prompt_for_skipped_boundary(candidate):
    return (
        "未进入VLM判定。\n"
        "该页边界没有形成可直接合并的文本候选，流程仅保存上下文截图用于人工核查。\n\n"
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
            "filename_pattern": "页号-页号_是否合并.png",
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
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross Page Text VLM Report</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f7f9; }}
header {{ padding: 20px 24px 12px; background: #fff; border-bottom: 1px solid #d9dee7; position: sticky; top: 0; z-index: 2; }}
h1 {{ margin: 0 0 10px; font-size: 22px; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 13px; }}
.summary span {{ background: #eef2f7; border: 1px solid #d9dee7; border-radius: 6px; padding: 6px 9px; }}
.tabs {{ margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap; }}
.tabs button {{ border: 1px solid #b8c0cc; background: #fff; border-radius: 6px; padding: 7px 11px; cursor: pointer; }}
.tabs button.active {{ background: #263445; color: #fff; border-color: #263445; }}
main {{ padding: 18px 24px 28px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
.card {{ background: #fff; border: 1px solid #d9dee7; border-radius: 8px; overflow: hidden; }}
.card.merge_true {{ border-left: 6px solid #168a4a; }}
.card.merge_false {{ border-left: 6px solid #b45309; }}
.card.merge_unknown {{ border-left: 6px solid #7b8794; }}
.card-header {{ padding: 12px 14px; border-bottom: 1px solid #e5e9f0; display: flex; justify-content: space-between; gap: 12px; }}
.pair {{ font-weight: 700; }}
.status {{ font-size: 12px; border-radius: 999px; padding: 3px 8px; background: #eef2f7; }}
.body {{ padding: 12px 14px; }}
img {{ width: 100%; height: auto; display: block; border: 1px solid #e5e9f0; border-radius: 6px; background: #fff; }}
.reason {{ margin: 10px 0 8px; line-height: 1.5; }}
.meta {{ font-size: 12px; color: #52606d; line-height: 1.45; }}
.prompt summary {{ margin-top: 8px; cursor: pointer; font-size: 12px; color: #1f5fbf; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #f3f5f8; border-radius: 6px; padding: 10px; font-size: 12px; }}
a {{ color: #1f5fbf; }}
@media (max-width: 980px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
<h1>跨页文本 VLM 判定报告</h1>
<div class="summary">
<span>总数 {summary["total"]}</span>
<span>合并 {summary["merge_true"]}</span>
<span>未合并 {summary["merge_false"]}</span>
<span>未知 {summary["merge_unknown"]}</span>
<span>模型 {html.escape(str(summary["model"]))}</span>
<span>提示词 {html.escape(str(summary["prompt_template"]))}</span>
</div>
<div class="tabs">
<button class="active" data-filter="all">全部</button>
<button data-filter="merge_true">合并</button>
<button data-filter="merge_false">未合并</button>
<button data-filter="merge_unknown">未知</button>
</div>
</header>
<main>
<div class="grid">
{cards}
</div>
<h2>summary.json</h2>
<pre>{summary_json}</pre>
</main>
<script>
const buttons = document.querySelectorAll(".tabs button");
const cards = document.querySelectorAll(".card");
buttons.forEach((button) => button.addEventListener("click", () => {{
  const filter = button.dataset.filter;
  buttons.forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  cards.forEach((card) => {{
    card.style.display = filter === "all" || card.dataset.status === filter ? "" : "none";
  }});
}}));
</script>
</body>
</html>
"""


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
        reason = f"未进入VLM判定: {skip_reason}"
    image_html = ""
    if record.get("marked_image"):
        src = _inline_image_src(report_dir, record["marked_image"]) or record["marked_image"]
        src = html.escape(src)
        image_html = f'<img src="{src}" alt="{html.escape(record["pair"])}">'
    prompt_html = ""
    if record.get("prompt_file"):
        prompt_text = _read_report_text(report_dir, record["prompt_file"])
        prompt_html = (
            f'<details class="prompt"><summary>提示词: {html.escape(record["prompt_file"])}</summary>'
            f'<pre>{html.escape(prompt_text)}</pre></details>'
        )
    return f"""<article class="card {html.escape(record["merge_status"])}" data-status="{html.escape(record["merge_status"])}">
<div class="card-header">
<div class="pair">{html.escape(record["pair"])}</div>
<div class="status">{_status_label(record["merge_status"])}</div>
</div>
<div class="body">
{image_html}
<div class="reason">理由: {html.escape(str(reason))}</div>
<div class="meta">置信度: {html.escape(confidence_text)}</div>
<div class="meta">A尾行: {html.escape(str(a_line))}</div>
<div class="meta">B首行: {html.escape(str(b_line))}</div>
{prompt_html}
</div>
</article>"""


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


def _status_label(merge_status):
    return {
        "merge_true": "合并",
        "merge_false": "未合并",
        "merge_unknown": "未知",
    }.get(merge_status, merge_status)


def _judge_with_lmstudio(candidate, prompt=None):
    image = candidate.get("image")
    if image is None:
        raise RuntimeError("page_images is required when no judge is injected")
    prompt = prompt or _prompt_for_candidate(candidate)
    pair = candidate.get("pair")
    return _call_model(prompt, image, pair=pair)


def _call_model(prompt, image, pair=None):
    model = _model_name()
    api_url = _api_url()
    api_key = _api_key()
    image_url = "data:image/png;base64," + _encode_image(image)
    payload = _make_chat_payload(model, prompt, image_url)
    errors = []
    request_start = time.time()
    logger.info("VLM model request begin: pair={}, model={}, api_url={}", pair, model, api_url)
    for string_image_url in (False, True):
        if string_image_url:
            payload = _make_chat_payload(model, prompt, image_url, string_image_url=True)
        for with_schema in (True, False):
            trial_payload = dict(payload)
            if not with_schema:
                trial_payload.pop("response_format", None)
            for attempt_index in range(len(REQUEST_RETRY_DELAYS_SECONDS) + 1):
                try:
                    trial_start = time.time()
                    logger.debug(
                        "VLM model request attempt: pair={}, attempt={}, schema={}, string_image_url={}",
                        pair,
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
                    logger.info(
                        "VLM model request success: pair={}, elapsed={}s, attempt_elapsed={}s, schema={}, string_image_url={}",
                        pair,
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
                            "VLM model request attempt failed without retry: pair={}, attempt={}, schema={}, string_image_url={}, error={}: {}",
                            pair,
                            attempt_index + 1,
                            with_schema,
                            string_image_url,
                            type(exc).__name__,
                            exc,
                        )
                        break
                    delay = REQUEST_RETRY_DELAYS_SECONDS[attempt_index]
                    logger.warning(
                        "VLM model request retry: pair={}, attempt={}, sleep={}s, schema={}, string_image_url={}, error={}: {}",
                        pair,
                        attempt_index + 1,
                        delay,
                        with_schema,
                        string_image_url,
                        type(exc).__name__,
                        exc,
                    )
                    time.sleep(delay)
    raise RuntimeError(" ; ".join(errors))


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


def _make_chat_payload(model, prompt, image_url, string_image_url=False):
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
        "response_format": SCHEMA,
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
