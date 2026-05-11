# VLM Cross-Page Feature Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current three-stage VLM cross-page merge judge with feature extraction, percent scoring, and deterministic normalization.

**Architecture:** Keep the existing candidate enumeration, merge application, report writer, and OpenAI-compatible request path. Change only the model prompts/schemas and the `_judge_with_lmstudio()` normalization path in `mineru/backend/utils/cross_page_text_vlm.py`, then update unit tests to lock the new behavior.

**Tech Stack:** Python 3.10-3.13, pytest, PIL test images, existing MinerU middle_json helpers.

---

## File Structure

- Modify `mineru/backend/utils/cross_page_text_vlm.py`
  - Replace the old layout/text/adjudication prompts with feature/scoring prompts.
  - Add `FEATURE_SCHEMA` and `SCORING_SCHEMA`.
  - Change `_judge_with_lmstudio()` to call `feature_extract` then `scoring`.
  - Replace `_normalize_adjudication_from_stage_decisions()` with `_normalize_scoring_from_features()`.
  - Keep final decision shape compatible with report and merge application: `is_continuation`, `confidence`, `target`, `a_last_line`, `b_first_line`, `joined_preview`, `b_starts_new_unit`, `same_list_item_body`, and `decision_basis`.
- Modify `tests/unittest/test_cross_page_text_vlm.py`
  - Replace old three-stage tests with two-stage tests.
  - Add hard-rule coverage for aligned merge, explicit-marker no-merge, and unclear-X conservative handling.
  - Keep report template coverage.

No new runtime files are needed.

---

### Task 1: Update Unit Tests For The New Two-Stage Contract

**Files:**
- Modify: `tests/unittest/test_cross_page_text_vlm.py`

- [ ] **Step 1: Replace the old stage-order test with a feature/scoring test**

Replace `test_vlm_judge_runs_layout_text_and_adjudication_stages` with:

```python
def test_vlm_judge_runs_feature_and_scoring_stages(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A tail."},
            "region_B_current_leaf": {"b_first_line": "B head"},
        },
        "image": Image.new("RGB", (12, 12), "white"),
    }
    calls = []

    def fake_call_model(prompt, image, pair=None, response_schema=None, stage=None):
        calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "pair": pair,
                "response_schema": response_schema,
                "image": image,
            }
        )
        if stage == "feature_extract":
            return {
                "marker_role": "none",
                "body_x": "aligned",
                "body_flow": "same",
            }
        if stage == "scoring":
            assert '"marker_role": "none"' in prompt
            assert '"body_x": "aligned"' in prompt
            return {
                "merge_score": 35,
                "new_unit_score": 65,
                "decision": "no_merge",
                "reason": "Scoring model was conservative, but hard rule should merge.",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert [call["stage"] for call in calls] == ["feature_extract", "scoring"]
    assert decision["is_continuation"] is True
    assert decision["target"] == "previous_leaf"
    assert decision["b_starts_new_unit"] is False
    assert decision["_stage_decisions"]["feature_extract"]["body_x"] == "aligned"
    assert decision["_stage_decisions"]["scoring"]["decision"] == "no_merge"
```

- [ ] **Step 2: Replace the old normalization test with explicit-marker no-merge coverage**

Replace `test_vlm_judge_normalizes_adjudication_from_stage_outputs` with:

```python
def test_vlm_judge_explicit_marker_role_forces_no_merge(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A closed sentence."},
            "region_B_current_leaf": {"b_first_line": "B new item"},
        },
        "image": Image.new("RGB", (12, 12), "white"),
    }

    def fake_call_model(prompt, image, pair=None, response_schema=None, stage=None):
        if stage == "feature_extract":
            return {
                "marker_role": "heading",
                "body_x": "aligned",
                "body_flow": "same",
            }
        if stage == "scoring":
            return {
                "merge_score": 95,
                "new_unit_score": 10,
                "decision": "merge",
                "reason": "Scoring model overrode marker evidence.",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert decision["is_continuation"] is False
    assert decision["target"] == "none"
    assert decision["b_starts_new_unit"] is True
    assert "explicit new structural marker" in decision["decision_basis"]
```

- [ ] **Step 3: Add a test for conservative handling when body-X is unclear**

Add this test below the explicit-marker test:

```python
def test_vlm_judge_unclear_body_x_does_not_merge(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A tail"},
            "region_B_current_leaf": {"b_first_line": "B head"},
        },
        "image": Image.new("RGB", (12, 12), "white"),
    }

    def fake_call_model(prompt, image, pair=None, response_schema=None, stage=None):
        if stage == "feature_extract":
            return {
                "marker_role": "none",
                "body_x": "unclear",
                "body_flow": "same",
            }
        if stage == "scoring":
            return {
                "merge_score": 92,
                "new_unit_score": 8,
                "decision": "merge",
                "reason": "Body flow looks continuous, but body-X is unclear.",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert decision["is_continuation"] is False
    assert decision["target"] == "none"
    assert decision["b_starts_new_unit"] is True
    assert "body-X is unclear" in decision["decision_basis"]
```

- [ ] **Step 4: Add a test that closed A still merges under the hard rule**

Add this test below the unclear-X test:

```python
def test_vlm_judge_closed_a_still_merges_when_marker_absent_and_body_x_aligned(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A complete sentence."},
            "region_B_current_leaf": {"b_first_line": "B continues the same body."},
        },
        "image": Image.new("RGB", (12, 12), "white"),
    }

    def fake_call_model(prompt, image, pair=None, response_schema=None, stage=None):
        if stage == "feature_extract":
            return {
                "marker_role": "none",
                "body_x": "aligned",
                "body_flow": "same",
            }
        if stage == "scoring":
            return {
                "merge_score": 40,
                "new_unit_score": 60,
                "decision": "unknown",
                "reason": "A is closed, but features support merge.",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert decision["is_continuation"] is True
    assert decision["target"] == "previous_leaf"
    assert decision["a_last_line"] == "A complete sentence."
    assert decision["b_first_line"] == "B continues the same body."
```

- [ ] **Step 5: Replace the prompt contract test**

Replace `test_prompts_define_same_item_body_and_body_x_rules` with:

```python
def test_prompts_define_general_marker_and_body_x_rules():
    assert "function, not concrete syntax" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "structural anchor" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "ignore that marker column" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "A's closed punctuation or complete sentence ending" in vlm.SCORING_PROMPT_TEMPLATE
    assert "0-100" in vlm.SCORING_PROMPT_TEMPLATE
```

- [ ] **Step 6: Run the updated tests and verify they fail before implementation**

Run:

```bash
pytest tests/unittest/test_cross_page_text_vlm.py -q
```

Expected: FAIL because `FEATURE_PROMPT_TEMPLATE`, `SCORING_PROMPT_TEMPLATE`, new stages, and new schemas do not exist yet.

- [ ] **Step 7: Commit the failing tests**

Run:

```bash
git add tests/unittest/test_cross_page_text_vlm.py
git commit -m "test: update VLM cross-page feature scoring contract"
```

Expected: commit succeeds with only the test file staged.

---

### Task 2: Implement Feature Extraction And Scoring Schemas

**Files:**
- Modify: `mineru/backend/utils/cross_page_text_vlm.py`

- [ ] **Step 1: Update the prompt version**

Change:

```python
PROMPT_VERSION = "vlm_layout_text_adjudicate_v2"
```

to:

```python
PROMPT_VERSION = "vlm_feature_scoring_v1"
```

- [ ] **Step 2: Replace the old prompt templates**

Delete `PROMPT_TEMPLATE`, `LAYOUT_FILTER_PROMPT_TEMPLATE`, `TEXT_CONTINUITY_PROMPT_TEMPLATE`, and `ADJUDICATION_PROMPT_TEMPLATE`.

Add these templates in their place:

```python
PROMPT_TEMPLATE = """You are a page-break MERGE judge for document reconstruction.
Return exactly one JSON object matching the schema. No markdown and no text outside JSON.

Data source note:
- The candidate text and boxes come from MinerU middle_json pdf_info[*].preproc_blocks.
- This is the original page-level block stream after OCR/image replacement and before para_blocks text/table merging.
- Do not infer from markdown or any already merged/deleted output.

This system uses a two-stage VLM chain:
1. feature_extract: observe compact marker, body-X, and body-flow features.
2. scoring: score merge and new-unit evidence from those features.

Python normalization applies deterministic hard rules after scoring.

Final output fields:
- is_continuation: boolean
- confidence: number from 0 to 1
- target: "previous_leaf" if true, otherwise "none"
- a_last_line: string
- b_first_line: string
- joined_preview: string, maximum 240 characters, no newline characters
- b_starts_new_unit: boolean
- same_list_item_body: boolean
- decision_basis: short reason consistent with is_continuation, maximum 240 characters

Metadata:
{metadata}
"""

FEATURE_PROMPT_TEMPLATE = """You are stage 1 of cross-page text merging: feature extraction.
Return exactly one JSON object matching the schema. Do not output any other text.

Task: observe compact features only. Do not make the final merge decision and do not include free-text evidence.

Feature rules:
- marker_role describes whether B starts with a new structural anchor and what role it has.
- Describe markers by function, not concrete syntax. A marker is any visually or textually distinct prefix at the beginning of B that establishes B as a separate document unit rather than body continuation.
- Use marker_role=\"none\" only when B clearly has no new explicit structural marker.
- Use marker_role=\"unclear\" when you cannot reliably decide whether B has a structural marker or what role it has.
- body_x compares A's body-text start X with B's body-text start X.
- If A has a marker, item identifier, bullet, clause prefix, or label prefix, ignore that marker column and compare the prose/body start X.
- body_flow describes whether A and B read like the same body flow.
- A can be textually closed and still have body_flow=\"same\" when B is a later sentence or paragraph in the same body unit.
- A's closed punctuation or complete sentence ending must not affect body_x.
- Page top position alone is not a structural marker.

Output fields:
- marker_role: "none" | "sibling" | "child" | "parent" | "heading" | "label" | "independent_item" | "unclear"
- body_x: "aligned" | "not_aligned" | "unclear"
- body_flow: "same" | "different" | "unclear"

Metadata:
{metadata}
"""

SCORING_PROMPT_TEMPLATE = """You are stage 2 of cross-page text merging: percent scoring.
Return exactly one JSON object matching the schema. Do not output any other text.

Task: score whether B should be appended directly to A using the feature extraction result and metadata.

Score semantics:
- merge_score is an integer 0-100 for how strongly B should be appended to A.
- new_unit_score is an integer 0-100 for how strongly B is a new structural unit.
- The two scores do not need to sum to 100.
- 0-30 means weak evidence, 31-69 means uncertain or mixed evidence, 70-89 means likely, and 90-100 means strong evidence.

Scoring rules:
- Explicit marker_role values other than none or unclear are strong new-unit evidence.
- marker_role=\"none\" plus body_x=\"aligned\" is the strongest merge evidence.
- body_flow is supporting evidence.
- body_x=\"unclear\" is insufficient for default merge.
- A's closed punctuation or complete sentence ending is weak evidence only, never a standalone no-merge reason.
- Do not override the extracted feature values. Score from them.

Output fields:
- merge_score: integer from 0 to 100
- new_unit_score: integer from 0 to 100
- decision: "merge" | "no_merge" | "unknown"
- reason: short reason, maximum 240 characters

Feature extraction result:
{features}

Metadata:
{metadata}
"""
```

- [ ] **Step 3: Replace old stage schemas with feature and scoring schemas**

Delete `LAYOUT_FILTER_SCHEMA` and `TEXT_CONTINUITY_SCHEMA`.

Add:

```python
FEATURE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hybrid_cross_page_feature_extract",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "marker_role": {
                    "type": "string",
                    "enum": [
                        "none",
                        "sibling",
                        "child",
                        "parent",
                        "heading",
                        "label",
                        "independent_item",
                        "unclear",
                    ],
                },
                "body_x": {"type": "string", "enum": ["aligned", "not_aligned", "unclear"]},
                "body_flow": {"type": "string", "enum": ["same", "different", "unclear"]},
            },
            "required": ["marker_role", "body_x", "body_flow"],
        },
    },
}

SCORING_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "hybrid_cross_page_scoring",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "merge_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "new_unit_score": {"type": "integer", "minimum": 0, "maximum": 100},
                "decision": {"type": "string", "enum": ["merge", "no_merge", "unknown"]},
                "reason": {"type": "string", "maxLength": 240},
            },
            "required": ["merge_score", "new_unit_score", "decision", "reason"],
        },
    },
}
```

- [ ] **Step 4: Update JSON schema validation to support integers**

In `_validate_model_json()`, after the number check, add:

```python
        if expected_type == "integer" and not isinstance(value, int):
            raise ValueError(f"model response field {field} must be integer")
```

Keep the existing `"number"` branch unchanged so final `SCHEMA` confidence still accepts int or float.

- [ ] **Step 5: Run schema/prompt test**

Run:

```bash
pytest tests/unittest/test_cross_page_text_vlm.py::test_prompts_define_general_marker_and_body_x_rules -q
```

Expected: PASS.

- [ ] **Step 6: Commit schemas and prompts**

Run:

```bash
git add mineru/backend/utils/cross_page_text_vlm.py
git commit -m "feat: add VLM cross-page feature scoring schemas"
```

Expected: commit succeeds with only `cross_page_text_vlm.py` staged.

---

### Task 3: Implement Two-Stage Judge And Deterministic Normalization

**Files:**
- Modify: `mineru/backend/utils/cross_page_text_vlm.py`

- [ ] **Step 1: Replace prompt helper functions**

Replace:

```python
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
```

with:

```python
def _feature_prompt_for_candidate(candidate):
    return FEATURE_PROMPT_TEMPLATE.format(
        metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2)
    )


def _scoring_prompt_for_candidate(candidate, features):
    return SCORING_PROMPT_TEMPLATE.format(
        features=json.dumps(features, ensure_ascii=False, indent=2),
        metadata=json.dumps(candidate["metadata"], ensure_ascii=False, indent=2),
    )
```

- [ ] **Step 2: Replace `_judge_with_lmstudio()`**

Replace the whole function with:

```python
def _judge_with_lmstudio(candidate, prompt=None):
    image = candidate.get("image")
    if image is None:
        raise RuntimeError("page_images is required when no judge is injected")
    pair = candidate.get("pair")
    feature_prompt = _feature_prompt_for_candidate(candidate)
    features = _call_model(
        feature_prompt,
        image,
        pair=pair,
        response_schema=FEATURE_SCHEMA,
        stage="feature_extract",
    )
    scoring_prompt = _scoring_prompt_for_candidate(candidate, features)
    scoring = _call_model(
        scoring_prompt,
        image,
        pair=pair,
        response_schema=SCORING_SCHEMA,
        stage="scoring",
    )
    decision = _normalize_scoring_from_features(candidate, features, scoring)
    decision["_stage_decisions"] = {
        "feature_extract": features,
        "scoring": scoring,
    }
    return decision
```

- [ ] **Step 3: Replace normalization implementation**

Delete `_normalize_adjudication_from_stage_decisions()`.

Add:

```python
NEW_UNIT_MARKER_ROLES = {
    "sibling",
    "child",
    "parent",
    "heading",
    "label",
    "independent_item",
}


def _normalize_scoring_from_features(candidate, features, scoring):
    marker_role = features.get("marker_role")
    body_x = features.get("body_x")
    scoring_decision = scoring.get("decision")
    merge_score = int(scoring.get("merge_score") or 0)
    new_unit_score = int(scoring.get("new_unit_score") or 0)

    if marker_role in NEW_UNIT_MARKER_ROLES:
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=False,
            confidence=max(_score_to_confidence(new_unit_score), 0.95),
            basis=f"Feature extraction found an explicit new structural marker: {marker_role}",
        )

    if marker_role == "none" and body_x == "aligned":
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=True,
            confidence=max(_score_to_confidence(merge_score), 0.9),
            basis="No structural marker and body-X is aligned",
        )

    if marker_role == "none" and body_x == "unclear":
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=False,
            confidence=max(_score_to_confidence(new_unit_score), 0.5),
            basis="No structural marker, but body-X is unclear; conservative no-merge",
        )

    if marker_role == "unclear":
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=False,
            confidence=max(_score_to_confidence(new_unit_score), 0.5),
            basis="Structural marker role is unclear; conservative no-merge",
        )

    if scoring_decision == "merge" and merge_score >= 70 and new_unit_score < 70:
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=True,
            confidence=_score_to_confidence(merge_score),
            basis=scoring.get("reason") or "Scoring stage selected merge",
        )

    if scoring_decision == "no_merge" or new_unit_score >= 70:
        return _decision_from_scoring(
            candidate,
            scoring,
            is_continuation=False,
            confidence=_score_to_confidence(max(new_unit_score, 70)),
            basis=scoring.get("reason") or "Scoring stage selected no_merge",
        )

    return _decision_from_scoring(
        candidate,
        scoring,
        is_continuation=False,
        confidence=max(_score_to_confidence(max(merge_score, new_unit_score)), 0.5),
        basis=scoring.get("reason") or "Scoring stage was uncertain; conservative no-merge",
    )
```

- [ ] **Step 4: Replace forced-decision helper**

Replace `_with_forced_decision()` with:

```python
def _decision_from_scoring(candidate, scoring, is_continuation, confidence, basis):
    metadata = candidate.get("metadata") or {}
    a_last_line = metadata.get("region_A_previous_leaf", {}).get("a_last_line") or ""
    b_first_line = metadata.get("region_B_current_leaf", {}).get("b_first_line") or ""
    joined_preview = f"{a_last_line} {b_first_line}".strip()
    return {
        "is_continuation": is_continuation,
        "confidence": min(max(float(confidence), 0), 1),
        "target": "previous_leaf" if is_continuation else "none",
        "a_last_line": a_last_line,
        "b_first_line": b_first_line,
        "joined_preview": joined_preview[:240],
        "b_starts_new_unit": not is_continuation,
        "same_list_item_body": False,
        "decision_basis": str(basis or scoring.get("reason") or "")[:240],
        "merge_score": scoring.get("merge_score"),
        "new_unit_score": scoring.get("new_unit_score"),
        "scoring_decision": scoring.get("decision"),
    }
```

Add this helper below `_decision_from_scoring()`:

```python
def _score_to_confidence(score):
    try:
        value = int(score)
    except (TypeError, ValueError):
        return 0
    return min(max(value, 0), 100) / 100
```

- [ ] **Step 5: Run the focused unit test suite**

Run:

```bash
pytest tests/unittest/test_cross_page_text_vlm.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit judge implementation**

Run:

```bash
git add mineru/backend/utils/cross_page_text_vlm.py
git commit -m "feat: use feature scoring for VLM cross-page text merge"
```

Expected: commit succeeds with only `cross_page_text_vlm.py` staged.

---

### Task 4: Final Verification And Cleanup

**Files:**
- Verify: `mineru/backend/utils/cross_page_text_vlm.py`
- Verify: `tests/unittest/test_cross_page_text_vlm.py`
- Verify: `docs/superpowers/specs/2026-05-11-vlm-cross-page-feature-scoring-design.md`

- [ ] **Step 1: Search for obsolete stage names in active code and tests**

Run:

```bash
rg -n "LAYOUT_FILTER|TEXT_CONTINUITY|ADJUDICATION|layout_filter|text_continuity|adjudication|new_marker_type|body_x_alignment" mineru/backend/utils/cross_page_text_vlm.py tests/unittest/test_cross_page_text_vlm.py
```

Expected: no matches in the target implementation or test file.

- [ ] **Step 2: Run lint-free diff check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Run the target unit tests**

Run:

```bash
pytest tests/unittest/test_cross_page_text_vlm.py -q
```

Expected: PASS.

- [ ] **Step 4: Review final git status**

Run:

```bash
git status --short
```

Expected: no tracked file modifications. Existing unrelated untracked files such as `.coverage`, `misc/`, or `uv.lock` may still appear and should not be staged unless the user explicitly asks.

- [ ] **Step 5: Commit final cleanup if needed**

If Step 1 or Step 2 required edits, run:

```bash
git add mineru/backend/utils/cross_page_text_vlm.py tests/unittest/test_cross_page_text_vlm.py
git commit -m "test: verify VLM cross-page feature scoring"
```

Expected: commit succeeds only if there were cleanup edits. If no edits were needed, skip this commit.

---

## Self-Review

Spec coverage:

- Feature schema fields are implemented in Task 2.
- Percent scoring schema and score semantics are implemented in Task 2.
- Two-stage VLM call sequence is implemented in Task 3.
- Deterministic hard rules are implemented in Task 3.
- Report compatibility is preserved by returning the old final decision fields and `_stage_decisions`.
- Required tests from the spec are covered in Task 1 and verified in Task 4.

Placeholder scan:

- No placeholder markers or deferred-work notes are present.
- Each code-changing step includes exact code snippets.
- Each verification step includes exact commands and expected outcomes.

Type consistency:

- Stage names are `feature_extract` and `scoring` in tests and implementation.
- Feature field names are `marker_role`, `body_x`, and `body_flow`.
- Scoring field names are `merge_score`, `new_unit_score`, `decision`, and `reason`.
- Final decision fields remain compatible with existing report and merge code.
