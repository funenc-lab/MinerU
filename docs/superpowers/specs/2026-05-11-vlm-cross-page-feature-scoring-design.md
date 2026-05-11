# VLM Cross-Page Feature Scoring Design

## Context

The current VLM cross-page text repair asks a model to judge whether the first text leaf on page B should be appended to the last text leaf on page A. It uses separate layout, text-continuity, and adjudication prompts, then normalizes some contradictory outputs in Python.

After changing VLM models, the structured decisions can become unstable. The new design reduces this dependency by separating observation, scoring, and deterministic normalization.

## Goal

Improve cross-page text merge stability for the hybrid VLM path.

The main merge definition is:

```text
If B has no new explicit structural marker, and B's body-text start X aligns
with A's body-text start X after ignoring A's marker or item identifier, B is
highly likely to be a continuation of A.
```

A closed sentence ending on A is not a hard no-merge signal. B can still belong to the same body flow when a page break splits a paragraph, numbered item body, clause body, or multi-sentence text unit.

## Non-Goals

- Do not change candidate generation in this design. The candidate remains the previous page's last text leaf and the current page's first text leaf.
- Do not add phrase-, label-, document-, or layout-specific special cases.
- Do not compute body-X alignment in Python. The VLM judges body-X alignment visually.
- Do not use free-text evidence in the feature extraction stage.

## Stage 1: Feature Extraction

The first VLM call only extracts compact structured features.

```json
{
  "marker_role": "none | sibling | child | parent | heading | label | independent_item | unclear",
  "body_x": "aligned | not_aligned | unclear",
  "body_flow": "same | different | unclear"
}
```

### Field Semantics

`marker_role` describes whether B starts with a new structural anchor and what role it has.

- `none`: B clearly has no new explicit structural marker.
- `sibling`: B starts a sibling unit relative to A.
- `child`: B starts a child unit under A or a nearby structure.
- `parent`: B starts a higher-level unit.
- `heading`: B starts a heading-like unit.
- `label`: B starts a label-like unit.
- `independent_item`: B starts an independent item that is not better classified above.
- `unclear`: the model cannot reliably decide whether B has a structural marker or what role it has.

Markers must be described by function, not concrete syntax. A marker is any visually or textually distinct prefix at the beginning of B that establishes B as a separate document unit rather than body continuation. It may be an ordered identifier, unordered bullet, clause label, heading label, item label, or another standalone structural anchor. The prompt should avoid enumerating specific marker spellings as rules.

`body_x` describes whether B's body-text start X aligns with A's body-text start X. If A has a marker, item identifier, bullet, or clause prefix, the model must ignore that marker column and compare the prose/body start X.

`body_flow` describes whether A and B read like the same body flow. A can be textually closed and still have `body_flow="same"` when B is a later sentence or paragraph in the same body unit.

## Stage 2: Scoring

The second VLM call receives the feature object and local A/B context, then returns percent scores and an explanation.

```json
{
  "merge_score": 0,
  "new_unit_score": 0,
  "decision": "merge | no_merge | unknown",
  "reason": "short reason"
}
```

### Score Semantics

`merge_score` is a 0-100 score for how strongly B should be appended to A.

`new_unit_score` is a 0-100 score for how strongly B is a new structural unit.

The scores do not need to sum to 100 because they measure different evidence directions.

Score bands:

```text
0-30    weak evidence
31-69   uncertain or mixed evidence
70-89   likely
90-100  strong evidence
```

The scoring prompt should treat:

- explicit `marker_role` values other than `none` or `unclear` as strong new-unit evidence;
- `marker_role="none"` plus `body_x="aligned"` as the strongest merge evidence;
- `body_flow` as supporting evidence;
- `body_x="unclear"` as insufficient for default merge;
- A's closed punctuation or complete sentence ending as weak evidence only, never a standalone no-merge reason.

`reason` is only for reporting and debugging. It must not drive program logic.

## Stage 3: Python Normalization

Python applies deterministic hard rules after scoring. These rules decide whether a merge is executed.

```text
if marker_role in {sibling, child, parent, heading, label, independent_item}:
    no_merge

elif marker_role == "none" and body_x == "aligned":
    merge

elif marker_role == "none" and body_x == "unclear":
    unknown/no_merge

elif marker_role == "unclear":
    unknown/no_merge

elif scoring.decision == "merge" and merge_score >= 70 and new_unit_score < 70:
    merge

elif scoring.decision == "no_merge" or new_unit_score >= 70:
    no_merge

else:
    unknown/no_merge
```

`unknown/no_merge` means the report should preserve an unknown status when possible, but no merge is executed.

The core behavior is:

```text
B has no new structural marker + body-X aligned => merge.
B has a clear new structural marker => no merge.
X unclear => conservative handling, do not default to merge.
```

## Code Mapping

Implementation should stay concentrated in `mineru/backend/utils/cross_page_text_vlm.py`.

Expected changes:

- Replace the current layout/text/adjudication chain with a feature/scoring chain.
- Add `FEATURE_PROMPT_TEMPLATE`.
- Add `FEATURE_SCHEMA`.
- Replace `ADJUDICATION_PROMPT_TEMPLATE` with `SCORING_PROMPT_TEMPLATE`.
- Add `SCORING_SCHEMA` with integer percent scores.
- Change `_judge_with_lmstudio()` to call:
  - `feature_extract`
  - `scoring`
  - normalization
- Replace `_normalize_adjudication_from_stage_decisions()` with a normalization function based on features and scoring.
- Keep report output compatible by storing stage decisions under `_stage_decisions`, using:
  - `feature_extract`
  - `scoring`

Candidate enumeration and merge application can remain unchanged.

## Testing

Update `tests/unittest/test_cross_page_text_vlm.py`.

Required coverage:

- `_judge_with_lmstudio()` calls `feature_extract` and `scoring` in order.
- `marker_role="none"` and `body_x="aligned"` forces merge, even if scoring says `no_merge`.
- explicit new-unit marker roles force no-merge, even if scoring says `merge`.
- `marker_role="none"` and `body_x="unclear"` does not execute a merge.
- a closed A ending does not prevent merge when `marker_role="none"` and `body_x="aligned"`.
- prompt/schema tests verify the generalized marker definition and body-X comparison rule.

## Open Decisions

None. The design intentionally chooses conservative handling for unclear marker or body-X states.
