from PIL import Image

from mineru.backend.utils import cross_page_text_vlm as vlm


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


def test_report_html_uses_external_templates(tmp_path):
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    html = vlm._render_report_html(
        {
            "total": 1,
            "merge_true": 1,
            "merge_false": 0,
            "merge_unknown": 0,
            "model": "test-model",
            "prompt_template": "prompt.txt",
        },
        [
            {
                "pair": "000-001",
                "merge_status": "merge_true",
                "decision": {
                    "confidence": 0.9,
                    "a_last_line": "A",
                    "b_first_line": "B",
                    "decision_basis": "Merged by test",
                },
            }
        ],
        report_dir,
    )

    assert "Cross Page Text VLM Report" in html
    assert "Reason: Merged by test" in html
    assert "Total 1" in html


def test_prompts_define_general_marker_and_body_x_rules():
    assert "function, not concrete syntax" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "structural anchor" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "ignore that marker column" in vlm.FEATURE_PROMPT_TEMPLATE
    assert "A's closed punctuation or complete sentence ending" in vlm.SCORING_PROMPT_TEMPLATE
    assert "0-100" in vlm.SCORING_PROMPT_TEMPLATE
