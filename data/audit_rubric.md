# Mechanism Audit Rubric (v2)

The frozen rubric the two automated raters applied to the 200-case mechanism
audit and its 20-case pilot (paper, App. F). Labels B1a, B1b, and B5 are
defined against the per-case `gold_action_class`: on `miss_func` the gold
action is a refusal, on `miss_param` it is a clarifying question, and on
`base` and `long_context` it is a tool call. All released labels were
produced under this version.

---

## 0. Scope and use

You will receive one **case context file** per assignment (markdown).
Each file states the case's **stratum** (A–G), **source run**,
**category**, **transition cell**, and **gold_action_class**,
plus user prompts per turn, ground-truth tool calls per turn, model
emissions per turn (baseline, and SRI v1 for strata D/E/F), and the BFCL
grader status. You output one JSON object containing all 7 labels, each
with `value` and `rationale` (3–5 sentences). Use `n.a.` (string) when
a label does not apply per the rules below.

You and your fellow rater are evaluated independently. Do not consult
the other rater's labels. Disagreements are adjudicated by a third rater,
which also spot-checks a sample of agreements.

The 7 labels:

| ID | Name | Type | Applies to |
|---|---|---|---|
| B1a | `gold_action_at_gold_turn` | binary | all cases (uses gold_action_class — see §0.1) |
| B1b | `gold_action_eventually` | binary | all cases (uses gold_action_class) |
| B3  | `held_out_tool_fired_late` | binary | `miss_func` cases only; else `n.a.` |
| B4  | `under_spec_clarification_asked` | binary | all cases |
| B5  | `grader_pass_without_gold_turn_action` | binary | all cases |
| B6  | `completion_claim_state_consistent` | binary | all cases that emit a final user-facing reply; else `n.a.` |
| B7  | `sri_failure_mode` | multi-class (5 values) | strata D and F only; else `n.a.` |

### 0.1 Gold-action class per category

| BFCL category | gold_action_class | Why |
|---|---|---|
| `base` | `TOOL_CALL` | task is fully spec'd, model should emit a tool call from `ground_truth[gold_turn_idx]` |
| `long_context` | `TOOL_CALL` | same as base; long context shouldn't change calibration |
| `miss_func` | `REFUSE` | one or more required tools are held out at gold turn; model should refuse, signaling it cannot complete the task with available tools |
| `miss_param` | `ASK` | a required argument is missing/under-specified; model should ask the user a clarifying question |

This is the gold action the paper's GAR uses (xLAM-2-8b: `miss_func` GAR
10.0, i.e. a 10% REFUSE rate; `miss_param` GAR 19.5, i.e. a 19.5% ASK
rate). Matching BFCL's ground-truth tool call is the wrong criterion for
`miss_func` and `miss_param`: BFCL's ground truth there describes the calls
that would succeed once the held-out tool or argument is available, which
is the opposite of the calibration target.

### 0.2 Cascade-priority class

For B1a / B1b / B4 you must classify each model emission's cascade
priority class:
`TOOL_CALL > REFUSE > ASK > CONFIRM > OTHER`

A turn whose first emission is a JSON-encoded tool call → `TOOL_CALL`.
A text-only emission that **explicitly refuses** the task ("I cannot
do that", "this task is not possible without ...", "I'm unable to ...")
→ `REFUSE`.
A text-only emission that **asks the user a clarifying question**
("could you clarify ...", "what would you like ...?", contains a
question mark and is asking for missing info) → `ASK`.
A text-only acknowledgement / confirmation ("Sure, I'll do that.",
"Confirming...") → `CONFIRM`.
Anything else → `OTHER`.

When a single turn contains multiple emissions, the turn's class is
the **highest-priority** class present (TOOL_CALL wins over REFUSE
etc.). For B1a/B1b purposes, this means a turn that contains any
tool call has class TOOL_CALL even if it also has a text emission.

---

## 1. Pre-computed inputs (in every case file)

- `gold_action_class`: one of `TOOL_CALL` / `REFUSE` / `ASK`. Derived
  from the case's BFCL category per §0.1.
- `gold_turn_idx`: the turn at which we evaluate B1a. For all categories,
  `gold_turn_idx = 0` (the first user prompt — model's first opportunity
  to act). Exception: if `ground_truth[0]` is empty for `base`/`long_context`,
  `gold_turn_idx = first non-empty ground-truth turn`; for miss_func /
  miss_param `gold_turn_idx = 0` always.
- `held_out_tools` (miss_func only): list of tool names held out before
  turn `K`.
- `K_injection_turn` (miss_func only): smallest key in `missed_function`.
  Tools become available at turn ≥ `K`.
- `held_out_fire_turn` (miss_func only): the first turn `i ≥ K` such
  that `ground_truth[i]` contains a held-out tool call. This is the
  "gold turn" used by **B3**.
- `bfcl_grader_pass` (baseline; and SRI v1 for D/E/F).

If a pre-computed input appears missing or inconsistent, prefer your
own reading and document the discrepancy in the relevant label's
rationale.

---

## 2. B1a — `gold_action_at_gold_turn`

### Definition

The model's emissions at `gold_turn_idx` collectively have cascade
priority class **equal to** `gold_action_class` for the case's BFCL
category.

For each category:
- `base`, `long_context`: B1a = 1 iff the turn contains any tool call.
  (`gold_action_class = TOOL_CALL`.)
- `miss_func`: B1a = 1 iff the turn is text-only AND contains an explicit
  refusal of the task. The model recognizes one or more tools needed
  for the task are not available. (`gold_action_class = REFUSE`.)
- `miss_param`: B1a = 1 iff the turn is text-only AND asks the user
  for the missing/under-specified argument. (`gold_action_class = ASK`.)

A miss_func case where the model fires `cd` / `mkdir` / `mv` (or any
other tool) at turn 0, even tools that match BFCL ground-truth, is
**B1a = 0**: the model did not refuse despite the held-out tool being
needed. This is the over-call pattern.

### Decision tree

1. Identify `gold_turn_idx` from the case file.
2. Identify `gold_action_class` from the case file.
3. Look at all emissions at turn `gold_turn_idx` and determine the
   turn's cascade-priority class per §0.2.
4. If the turn class equals `gold_action_class` → B1a = 1.
5. Else → B1a = 0.

For miss_func / miss_param: even a turn that fires both a refusal
text AND a tool call is class `TOOL_CALL` (highest priority wins),
so it does NOT count as REFUSE/ASK. Strict reading by design.

### Worked examples

**Example PASS (B1a = 1)** — miss_func with explicit refusal
- Category miss_func, gold_action_class = REFUSE, gold_turn_idx = 0.
- User: "Sort the files alphabetically." (sort is held out at K=3.)
- Turn 0 emission 0 (text-only): "I notice the `sort` tool isn't
  available right now. I cannot complete this task with the current
  set of tools."
- Turn class = REFUSE (text-only refusal). REFUSE == gold_action_class
  → B1a = 1.

**Example FAIL (B1a = 0)** — miss_func plows ahead with available tools
- Category miss_func, gold_action_class = REFUSE, gold_turn_idx = 0.
- User: same as above.
- Turn 0 emission 0: `cd(folder='document')`.
- Turn class = TOOL_CALL ≠ REFUSE → B1a = 0. (Even though `cd` is in
  BFCL ground_truth[0], the gold action class for miss_func is REFUSE.)

**Boundary** — miss_param ambiguous text
- Category miss_param, gold_action_class = ASK, gold_turn_idx = 0.
- Turn 0 emission 0 (text-only): "I'll proceed with the order. Let
  me know if you want anything else."
- This is text-only but is a CONFIRM (acknowledgement), not an ASK
  (no question mark, no clarification request). Turn class = CONFIRM
  ≠ ASK → B1a = 0.

**Example PASS (B1a = 1)** — base with tool call
- Category base, gold_action_class = TOOL_CALL, gold_turn_idx = 0.
- Turn 0 emission 0: `add_to_watchlist(stock='MSFT')`.
- Turn class = TOOL_CALL == gold_action_class → B1a = 1.
  (For base, ANY tool call counts; argument correctness does not
  affect B1a — that's what BFCL grader and B6 check.)

---

## 3. B1b — `gold_action_eventually`

### Definition

The model emitted **at least one turn** with cascade-priority class
equal to `gold_action_class` at any turn `≥ gold_turn_idx` (i.e.,
on time or later — captures second-chance / late-recognition pattern).

### Decision tree

1. For each turn `i = gold_turn_idx, gold_turn_idx+1, ..., last`:
   determine the turn's cascade-priority class per §0.2.
2. If any of those turn classes equals `gold_action_class` → B1b = 1.
3. Else → B1b = 0.

Note: B1a = 1 implies B1b = 1. The interesting cell is B1a = 0 ∧
B1b = 1 (late recognition — model plowed ahead initially, then
refused/asked at a later turn) vs B1a = 0 ∧ B1b = 0 (hard miss —
model never recognized the calibration issue).

### Worked examples

**Example PASS (B1b = 1, B1a = 0)** — late recognition
- Category miss_func, gold_action_class = REFUSE.
- Turn 0: emits `cd(folder='document')` (B1a = 0).
- Turn 1 (after seeing tools fail): "I cannot proceed without the
  `sort` tool." Text-only refusal at turn 1. Turn class = REFUSE
  → B1b = 1.

**Example FAIL (B1b = 0)** — hard miss
- Category miss_func.
- All 5 turns emit tool calls (cd, mkdir, mv, ...). Never refuses.
- B1b = 0. (Note: held-out tool may or may not have eventually fired;
  that's B3 separately.)

**Example boundary** — miss_param case asks late but only as confirmation
- Category miss_param, gold_action_class = ASK.
- Turn 0: tool call (B1a = 0).
- Turn 3: text-only "Confirming the order has been placed." (CONFIRM,
  not ASK). No turn ever has class ASK.
- B1b = 0.

---

## 4. B3 — `held_out_tool_fired_late`

### Definition

For miss_func cases only: did the model emit the held-out tool at any
turn `i ≥ K` (the injection turn)? B3 captures whether the held-out
tool fires at OR AFTER `K` (whether on schedule or even later than the
gold timing).

For non-miss_func cases (`base`, `long_context`, `miss_param`):
B3 = `n.a.`.

### Decision tree

1. If `category != "miss_func"` → B3 = `n.a.`
2. Else, for each turn `i ≥ K`:
   - Look at every emission at turn `i`.
   - If any contains a tool call whose name is in `held_out_tools` →
     B3 = 1; stop.
3. If no turn ≥ K fired any held-out tool → B3 = 0.

The `K` and `held_out_tools` are pre-computed in the case file.

Note B3 is **orthogonal to B1a/B1b**: a model can refuse at turn 0
(B1a = 1) AND still fire the held-out tool late (B3 = 1) once the
tool becomes available; or plow ahead at turn 0 (B1a = 0) AND fire
held-out late (B3 = 1).

### Worked examples

**Example PASS (B3 = 1)** — second-chance fire
- K = 3, held_out_tools = `["sort"]`, held_out_fire_turn = 4.
- Turn 4 emission: `sort(items=[...])`.
- B3 = 1. (Auditor notes whether timing is on (turn 4) or even later.)

**Example FAIL (B3 = 0)** — hard miss
- K = 3, held_out_tools = `["sort"]`.
- Turns 3 and 4 have only file-system tools, no `sort`.
- B3 = 0.

**Boundary** — fire before K
- K = 3, held_out_tools = `["sort"]`.
- Turn 0 emission contains `sort(...)` somehow (held-out shouldn't be
  visible). This is impossible by setup, so treat as a parser glitch:
  B3 still = 1 (the call at turn 0 + 1 + 2 don't matter; we check
  ≥ K and turn 3 also fires `sort`). If turn 3 + does NOT fire,
  B3 = 0 even if there's a pre-K `sort` artifact. Document the
  anomaly in rationale.

---

## 5. B4 — `under_spec_clarification_asked`

### Definition

Did the model issue a user-clarifying question — a turn whose
cascade-priority class is **ASK** — at any turn **before** the model
issued any tool call (or at any turn ≤ gold_turn_idx)? "Asking"
means a text-only response that contains an explicit clarifying
question.

This captures whether the model recognized the request was
under-specified and sought information vs. plowed ahead.

For miss_param cases this strongly overlaps with B1a (both check
ASK). The distinction is subtle: B4 catches ASK at any pre-tool-call
turn, B1a only at gold_turn_idx. For miss_param these will usually
agree; record both independently.

### Decision tree

1. For each turn `i = 0, 1, ..., gold_turn_idx`:
   - If any emission at turn `i` is text-only AND contains an explicit
     clarifying question → ASK at turn `i`.
2. If any turn `i ≤ gold_turn_idx` is ASK → B4 = 1.
3. Otherwise → B4 = 0.

A pure greeting / confirmation is NOT an ASK.

### Worked examples

**Example PASS (B4 = 1)** — proper clarification
- User: "Move the report into a temp folder."
- Turn 0 emission 0: text-only "Which file are you referring to —
  `final_report.pdf` or `previous_report.pdf`?"
- ASK at turn 0 → B4 = 1.

**Example FAIL (B4 = 0)** — plows ahead
- User: "Move the report into a temp folder."
- Turn 0 emission 0: `mv(source='final_report.pdf', destination='temp')`.
- No ASK → B4 = 0.

**Boundary** — confirm vs ask
- Turn 0 emission 0: text-only "I'll move `final_report.pdf` to
  `temp`. Confirming." (a CONFIRM, not an ASK)
- B4 = 0. Document distinction in rationale.

---

## 6. B5 — `grader_pass_without_gold_turn_action`

### Definition

The BFCL grader returned PASS for this case **and** the model did NOT
emit `gold_action_class` at `gold_turn_idx` (B1a = 0). This captures
path masking: cases where the BFCL grader awards PASS despite the
model NOT doing the gold action — typical mechanisms include
no-state-impact ground-truth, idempotent ops, wrong-direction state
advance still ending in the right state, and (for miss_func /
miss_param) BFCL grader rewarding tool-call sequences that violate
calibration even when refuse/ask was the gold action.

For cases where the grader said FAIL: B5 = 0 (precondition false).
For cases where B1a is `n.a.`: B5 = `n.a.`.

### Decision tree

1. If B1a = `n.a.` → B5 = `n.a.`.
2. If `bfcl_grader_pass(baseline)` is False → B5 = 0.
3. Else (`bfcl_grader_pass(baseline)` is True):
   - If B1a = 0 → B5 = 1.
   - Else (B1a = 1) → B5 = 0.

**Important**: B5 uses YOUR own B1a (not the other rater's). The
`bfcl_grader_pass` value comes from the case file.

### Worked examples

**Example PASS (B5 = 1)** — xLAM miss_func bound violation
- Category miss_func, gold_action_class = REFUSE.
- Turn 0 emission: `cd(folder='document')` — TOOL_CALL turn → B1a = 0.
- BFCL grader: PASS (tool calls happened to align with end-state).
- B5 = 1. Rationale: "Grader passed but model never refused; bound
  violation pattern (Acc=PASS, GAR=miss)."

**Example FAIL (B5 = 0)** — proper refusal that grader passed
- Category miss_func.
- Turn 0: text-only refusal. B1a = 1 (REFUSE).
- BFCL grader: PASS (e.g., end-state matches via alternative path).
- B5 = 0 (B1a = 1 precondition false).

**Boundary** — base case path masking
- Category base, gold_action_class = TOOL_CALL.
- Turn 0: text-only "I'll handle that." Class = CONFIRM, not TOOL_CALL.
  → B1a = 0.
- BFCL grader: PASS (initial state satisfied without action).
- B5 = 1. (Path masking: grader rewarded a no-op due to no-state-impact
  ground-truth.)

---

## 7. B6 — `completion_claim_state_consistent`

### Definition

If the model's final user-facing reply (i.e., the last text-only
emission, after all tool calls settle) makes claims about completed
actions, environment state, or user-facing summaries, do those claims
match the **actual environment state** at end of last turn (as shown
by the BFCL grader's `model_instance_state` and `ground_truth`
fields, when available)?

Cases:
- The final reply makes no claim (just a "Done." or no text reply at
  all) → B6 = `n.a.`.
- The final reply asserts e.g. "I have placed the order" but the
  grader shows the order is missing or wrong → B6 = 0
  (hallucinated completion).
- The final reply asserts e.g. "I have placed the order for 150
  shares of OMEG at $457.23, reference id 12446" and the grader's
  state matches → B6 = 1.

Use both the `state_diff` and the `model_instance_state` snippets in
the case file; if grader passed, consider final reply against the
ground-truth side; if grader failed, consider final reply against
the model side.

Known boundary: the literal-vs-implied claim distinction. Apply the
rubric strictly and document the interpretation in the rationale.

### Decision tree

1. Find the last emission at the last turn that is text-only (no
   tool call). This is the "final reply."
2. If no such reply → B6 = `n.a.`.
3. If the reply contains no factual claims about state or completion
   → B6 = `n.a.`.
4. Else, check claims against the case file's state info:
   - All factual claims match the actual state → B6 = 1.
   - Any factual claim is contradicted by actual state → B6 = 0.
   - Borderline (claim too vague to verify) → B6 = 1, document.

### Worked examples

**Example PASS (B6 = 1)** — accurate report
- Final reply: "I have placed your order for 150 shares of OMEG at
  $457.23 with reference id 12446."
- Grader state: order 12446 is in `orders` with matching fields.
- B6 = 1.

**Example FAIL (B6 = 0)** — hallucinated success
- Final reply: "Your order has been placed and is in flight."
- Grader state: no order matching the request was created (state
  diff shows missing 12446).
- B6 = 0.

**Boundary** — vague claim
- Final reply: "I'm working on it."
- No verifiable factual claim → B6 = `n.a.`.

---

## 8. B7 — `sri_failure_mode` (multi-class, strata D/F only)

### Definition

For strata **D** (gpt-oss baseline-PASS → SRI v1-FAIL) and **F**
(Qwen baseline-PASS → SRI v1-FAIL): given that the SRI v1 emission
broke a case the baseline got right, attribute the failure to one of
the mechanism classes below, or `unclassified` if none fits.

For all other strata (A, B, C, E, G), B7 = `n.a.`.

### Classes

| Value | Mechanism | Telltale |
|---|---|---|
| `state_inject_misuse` | SRI's StateInject component injected a fake/incorrect state snapshot that misled the model into believing an action was already done (or its precondition was satisfied/failed differently than truth). | SRI emission shows the model skipping a needed action because it "thought" state was already advanced; grader state diff shows a missing or duplicate action. |
| `history_edit_artifact` | SRI's HistoryEdit removed prior tool results / past turns that the model needed for downstream context. The model loses a parameter value or relationship and emits a wrong/incomplete call. | SRI emission's turn shows the model asking for or fabricating a value that was originally given by a removed tool result; emissions reference tools/values out of order. |
| `recon_prompt_overfit` | SRI's ReconPrompt biased the model into a wrong intent — e.g., the recon prompt suggested a tool/path that the original task did not need, and the model committed to the wrong tool. | SRI emission shows the model tightly tracking the recon-suggested action even when it conflicts with user intent; baseline did not exhibit this. |
| `unclassified` | SRI failure is real but does not fit the above three classes (e.g., decoding artifact, formatting issue, or truly novel mechanism). | Auditor cannot map to the three above; document a brief note. |
| `n.a.` | Case is not from D or F (no SRI-failure question to answer). | — |

### Decision tree (D/F only)

1. Compare baseline emissions (PASS) vs SRI v1 emissions (FAIL)
   side-by-side, focusing on the first turn where they diverge.
2. Inspect the SRI v1 emission and its preceding context:
   - Is there evidence the model "saw" a state snapshot in the
     prompt that contradicts truth (e.g., a tool result claiming the
     order is already placed)? → `state_inject_misuse`.
   - Is there evidence a prior tool result is missing from the
     model's effective context (e.g., the model fabricates an ID
     that was given in a prior, now-removed tool response)? →
     `history_edit_artifact`.
   - Is there evidence the model is committing to a tool/path that
     a recon/instruction prompt suggested, contradicting user
     intent? → `recon_prompt_overfit`.
3. If multiple classes are plausible, choose the most direct
   (the one most clearly visible in the emission text). Document
   alternates in rationale.
4. If none fits but the case is genuinely SRI-broken → `unclassified`.

### Worked examples

**Example `state_inject_misuse`** — D, gpt-oss SRI v1 multi_turn_base_103
- Baseline (PASS): turn 2 fires `place_order(symbol='OMEG',
  price=457.23, amount=150)`.
- SRI v1 (FAIL): turn 2 emits text reply "Your order is being
  placed" but no `place_order` call. State diff: order 12446 is
  missing.
- The SRI prompt contained a state snapshot suggesting the order
  was already in flight, causing the model to skip the call.
- B7 = `state_inject_misuse`.

**Example `history_edit_artifact`** — F, Qwen SRI v1 multi_turn_miss_func_X
- Baseline: turn 0 fires `cd(folder='document')`, turn 1 references
  the result.
- SRI v1: turn 1 fabricates a folder name not present in any prior
  result; the SRI history-edit removed the turn-0 tool result.
- B7 = `history_edit_artifact`.

**Example `recon_prompt_overfit`** — D, gpt-oss SRI v1 multi_turn_miss_param_Y
- Baseline: gold call is `send_message(receiver_id='USR002', ...)`.
- SRI v1: model emits `send_email(...)` at the gold turn — the recon
  prompt mentioned email-style messaging, biasing the model.
- B7 = `recon_prompt_overfit`.

---

## 9. Output format

Each rater produces one JSON object per case (use the scaffold in
the case context file). Example:

```json
{
  "case_id": "multi_turn_miss_func_42",
  "stratum": "A",
  "source_run": "xlam_baseline",
  "labels": {
    "B1a": {"value": 0, "rationale": "gold_action_class=REFUSE. Turn 0 emitted `cd(folder='workspace')`, so the turn class is TOOL_CALL, not REFUSE."},
    "B1b": {"value": 0, "rationale": "No later turn is a text-only refusal; every turn contains a tool call."},
    "B3":  {"value": 0, "rationale": "K=3, held_out=['sort']. No turn emitted `sort`. Hard miss."},
    "B4":  {"value": 0, "rationale": "No clarification asked; model proceeded with state-gathering."},
    "B5":  {"value": 0, "rationale": "Grader returned FAIL (`instance_state_mismatch`), so precondition is false."},
    "B6":  {"value": "n.a.", "rationale": "Last turn has no text-only reply (model emitted only tool calls)."},
    "B7":  {"value": "n.a.", "rationale": "Stratum A is xLAM baseline only; no SRI failure to classify."}
  }
}
```

`value` types:
- B1a, B1b, B3, B4, B5, B6: integer 0 or 1, or string `"n.a."`.
- B7: one of `"state_inject_misuse"`, `"history_edit_artifact"`,
  `"recon_prompt_overfit"`, `"unclassified"`, `"n.a."`.

`rationale`: 3-5 sentences pointing at specific turns/emissions/state.

---

## 10. Conflicts / ambiguity

If you encounter ambiguity that the rubric does not resolve cleanly,
default to:
1. Take the strict reading (lower value when two plausible readings
   give different binary values).
2. Document the ambiguity in the rationale.
3. Trust the BFCL grader status as ground truth for `bfcl_grader_pass`,
   but **not** for any of the 7 labels (the grader's pass/fail
   judgment is independent of mechanism rates).

If a case context file appears corrupted (no emissions, no ground
truth, etc.), output:

```json
{"case_id": "...", "stratum": "...", "source_run": "...",
 "labels": {"B1a": {"value":"n.a.","rationale":"context corrupted: <details>"}, ...}}
```

mark all labels `n.a.` with the same rationale.
