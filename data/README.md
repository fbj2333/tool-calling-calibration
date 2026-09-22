# Released data

Three datasets, one JSONL file each (UTF-8, one record per line), plus the
audit rubric.

| File | What it is | Paper |
|---|---|---|
| `audit_labels.jsonl` | 220 audited BFCL cases: the case context, both raters' labels, adjudication, and final labels | App. F |
| `audit_rubric.md` | the frozen rubric (v2) both raters applied | App. F |
| `tau2_confirm_judgments.jsonl` | 1,887 database writes on τ²-bench (1,879 with a judge verdict) | Tab. 3 |
| `tau2_ask_label_audit.jsonl` | the 41 τ²-bench ASK candidate tasks, each checked against its task text | Tab. 2 (ASK) |

## `audit_labels.jsonl`: mechanism audit

200 cases (`split = "main"`) plus the 20-case pilot (`split = "pilot"`), all
labelled under `audit_rubric.md`. Cases are BFCL multi-turn trajectories in
native function-calling mode, drawn from seven strata:

| Stratum | Model | Categories | Cases (main) |
|---|---|---|---|
| A | xLAM-2-8b-fc-r | miss_func | 48 |
| B | xLAM-2-8b-fc-r | miss_param | 30 |
| C | Qwen3-8B | all four; FAIL at baseline and under SRI v1 | 40 |
| D | gpt-oss-20b | all four; PASS at baseline, FAIL under SRI v1 | 30 |
| E | Qwen3-8B | all four; FAIL at baseline, PASS under SRI v1 | 20 |
| F | Qwen3-8B | all four; PASS at baseline, FAIL under SRI v1 | 22 |
| G | gpt-oss-20b | all four; FAIL at baseline and under SRI v1 | 10 |

| Field | Meaning |
|---|---|
| `split` | `main` or `pilot` |
| `case_id` | BFCL test-entry id |
| `model` | checkpoint whose trajectory is audited |
| `category` | `base`, `long_context`, `miss_func`, or `miss_param` |
| `stratum` | A–G (table above) |
| `transition` | strata C–G: the BFCL outcome at baseline and under SRI v1, e.g. `PASS_to_FAIL`; `null` for A and B |
| `context` | the case file the raters labelled (markdown): prompts, ground truth, model emissions, grader status, pre-computed rubric inputs |
| `rater_a`, `rater_b` | per label (`B1a` … `B7`): `{"value", "rationale"}` |
| `adjudication` | `main` only: labels the two raters disagreed on, `{"value", "note"}` |
| `spot_check` | `main` only: agreements sampled for review, `{"value", "correct", "note"}`; otherwise `null` |
| `final` | `main` only: the value per label after adjudication; `null` for the pilot |

Values are `0`, `1`, or `"n.a."`; B7 takes one of the class names in the
rubric. Two independent automated raters label every case; a third rater
adjudicates each disagreement in the main split and spot-checks a sample of
agreements. All three are separate Claude Opus 4.7 sessions.

`python scripts/audit_stats.py` recomputes the App. F tables from this file:
per-label Cohen's κ and agreement before adjudication, post-adjudication rates
with Wilson intervals, and per-stratum agreement.

## `tau2_confirm_judgments.jsonl`: confirmation before database writes

One record per database write made by a model on τ²-bench airline or retail.
τ²-bench's written policy requires explicit user confirmation before every
write; an LLM judge (gpt-5.4-2026-03-05) read the dialogue before each write
and decided whether that confirmation was obtained
(`scripts/judge_tau2_confirm.py`).

| Field | Meaning |
|---|---|
| `model`, `domain` | agent model; `airline` or `retail` |
| `sim_id` | τ²-bench simulation id |
| `tool` | the write tool called |
| `confirmed` | judge verdict: `true` / `false`; `null` when the judge returned no verdict (8 writes, excluded from Tab. 3) |
| `passed` | whether the trajectory containing the write passed τ²-bench's reward |
| `judge_reasoning` | the judge's one-line justification |

GAR is the share of judged writes with `confirmed = true`; Acc is the share of
the same writes whose trajectory passed. `python scripts/confirm_table.py`
prints Tab. 3.

## `tau2_ask_label_audit.jsonl`: ASK labels on τ²-bench

τ²-bench has no ASK annotation. `scripts/classify_tau2_tasks.py` selects
candidate tasks by keyword over each task's `user_scenario`; each of the 41
candidates was then checked by reading the full task text.

| Field | Meaning |
|---|---|
| `domain`, `task_id` | τ²-bench task |
| `label` | `valid` (the scenario withholds a task-relevant detail until the agent asks), `ambiguous`, or `false_positive` (the matched text is not a detail the agent needs to ask for, e.g. a confirmation step or a detail the user must not reveal) |
| `in_paper_ask_subset` | whether the task is in the ASK subset of Tab. 2 (8 airline / 28 retail) |
| `evidence` | the matched scenario text |
| `reason` | one-sentence justification |

The Tab. 2 subset keeps every candidate except five airline tasks (8, 29, 33,
34, 38) in which the matched text gives the agent nothing to ask about: the
detail is already stated, or "asks" refers to a fee demand. The tasks labelled
`valid` form a stricter subset (6 airline / 13 retail).
`scripts/tau2_ask_subset.py` reports ASK GAR and Acc on all candidates, on the
Tab. 2 subset, and on the strict subset.
