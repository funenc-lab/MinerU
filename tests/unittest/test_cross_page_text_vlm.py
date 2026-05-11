from PIL import Image

from mineru.backend.utils import cross_page_text_vlm as vlm


def test_vlm_judge_runs_layout_text_and_adjudication_stages(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A tail,"},
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
        if stage == "layout_filter":
            return {
                "body_x_alignment": "aligned",
                "b_has_new_paragraph_start": False,
                "b_has_new_list_or_clause_marker": False,
                "new_marker_type": "none",
                "b_starts_new_block_by_layout": False,
                "evidence": "Body X is approximately aligned and B has no new marker",
            }
        if stage == "text_continuity":
            return {
                "a_is_textually_unclosed": True,
                "b_completes_a": True,
                "same_body_flow": True,
                "continuation_type": "unfinished_phrase",
                "evidence": "A ends with a comma and B completes the same body",
            }
        if stage == "adjudication":
            assert '"body_x_alignment": "aligned"' in prompt
            assert '"same_body_flow": true' in prompt
            return {
                "is_continuation": True,
                "confidence": 0.95,
                "target": "previous_leaf",
                "a_last_line": "A tail,",
                "b_first_line": "B head",
                "joined_preview": "A tail, B head",
                "b_starts_new_unit": False,
                "same_list_item_body": True,
                "decision_basis": "Layout is aligned and text is continuous",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert [call["stage"] for call in calls] == [
        "layout_filter",
        "text_continuity",
        "adjudication",
    ]
    assert decision["is_continuation"] is True
    assert decision["_stage_decisions"]["layout_filter"]["body_x_alignment"] == "aligned"
    assert decision["_stage_decisions"]["text_continuity"]["same_body_flow"] is True


def test_vlm_judge_normalizes_adjudication_from_stage_outputs(monkeypatch):
    candidate = {
        "pair": "000-001",
        "metadata": {
            "prev_page_idx": 0,
            "current_page_idx": 1,
            "region_A_previous_leaf": {"a_last_line": "A tail involve"},
            "region_B_current_leaf": {"b_first_line": "B head"},
        },
        "image": Image.new("RGB", (12, 12), "white"),
    }

    def fake_call_model(prompt, image, pair=None, response_schema=None, stage=None):
        if stage == "layout_filter":
            return {
                "body_x_alignment": "shifted",
                "b_has_new_paragraph_start": True,
                "b_has_new_list_or_clause_marker": False,
                "new_marker_type": "none",
                "b_starts_new_block_by_layout": True,
                "evidence": "B looks like a new paragraph but has no new marker",
            }
        if stage == "text_continuity":
            return {
                "a_is_textually_unclosed": True,
                "b_completes_a": False,
                "same_body_flow": False,
                "continuation_type": "none",
                "evidence": "A is unclosed",
            }
        if stage == "adjudication":
            return {
                "is_continuation": False,
                "confidence": 0.9,
                "target": "none",
                "a_last_line": "A tail involve",
                "b_first_line": "B head",
                "joined_preview": "A tail involve B head",
                "b_starts_new_unit": True,
                "same_list_item_body": False,
                "decision_basis": "Final model incorrectly chose a new block",
            }
        raise AssertionError(f"unexpected stage: {stage}")

    monkeypatch.setattr(vlm, "_call_model", fake_call_model)

    decision = vlm._judge_with_lmstudio(candidate)

    assert decision["is_continuation"] is True
    assert decision["target"] == "previous_leaf"
    assert decision["b_starts_new_unit"] is False


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


def test_prompts_define_same_item_body_and_body_x_rules():
    assert "A page break may split the body of one numbered/list item" in (
        vlm.TEXT_CONTINUITY_PROMPT_TEMPLATE
    )
    assert "A clause/list number at the beginning of A does not by itself make A a heading" in (
        vlm.TEXT_CONTINUITY_PROMPT_TEMPLATE
    )
    assert "compare the prose body-text start x" in vlm.LAYOUT_FILTER_PROMPT_TEMPLATE.lower()
    assert "same numbered/list item" in vlm.ADJUDICATION_PROMPT_TEMPLATE
