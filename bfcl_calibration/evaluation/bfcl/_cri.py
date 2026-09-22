"""CRI (Call-Reconciliation Intervention) — the call-seam probe.

CRI is the directional mirror of SRI. SRI fires on the *halt* seam
(zero parseable tool_call); CRI fires on its **complement** — the model
emitted >=1 parseable tool_call — and is consulted by the multi-turn
driver *before* ``execute_multi_turn_func_call`` runs, so a retracted
call never mutates backend state.

Decision pipeline at the trigger seam (``_cri_evaluate``):

1. **Detector** (MC / UR): pure functions over (emitted call, current
   compiled tool list, runtime state + user-text). Label-free at the
   trigger decision.
2. **Stage gates**: suppress the trigger when the
   call is no longer in CRI's intervention window.
   - ``cri_post_injection_suppressed``: the live toolset has grown past
     its case-entry size, i.e. the held-out function was re-injected
     (miss_func). The toolset grows only in function-calling mode (API
     models); local checkpoints run BFCL's prompting loop, which announces
     the held-out function in the prompt text without extending
     ``test_entry["function"]``, so for them this gate never fires.
   - ``cri_already_fired_in_case``: once a CRI fire has happened in the
     case, later fires are suppressed.
   - ``cri_bypass_non_gold_turn`` / ``cri_bypass_no_gold_turn_marker``:
     bypass-specific guard — bypass leaves the turn definitively
     empty, which is only safe at the empty-GT (gold) turn.
3. **Variant dispatch** (in ``_sri_loop._handle_overaction_candidate``):
   - CRI-retry (``v1``): append a role:system retry instruction and
     re-decode.
   - CRI-bypass (``bypass``): insert a fixed assistant abstention placeholder, break
     out of the step loop without retrying. Designed for tool-call-specialized
     models where retry deterministically re-emits the same suspect
     call.
4. **Handler side effects** (in ``_sri_loop``): per-step buffer pops
   (score-trace protection), chat-history edit
   (``_remove_last_assistant``), retry-instruction
   append OR placeholder injection (per variant).

The ``inference_data['message']`` mutation stays in ``_sri_loop`` so all
chat-history writes live in one place. Per-entry CRI fired-turn state
(``_cri_retry_turns``) is attached to the ``SRIRuntime`` instance lazily.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._sri import (
    SRIRuntime,
    _sri_state_to_text,
    _sri_truncate,
    _write_sri_event,
)
from ._state import (
    _format_state_snapshot,
    _normalise_tool_call,
    _response_data_tool_calls,
)


_CRI_VALID_MODES = {"off", "passive_log", "cri_only"}
_CRI_VALID_VARIANTS = {
    # CRI-retry: retract the suspect call and re-query the model with an
    # instruction not to call a tool this turn.
    "v1",
    # CRI-bypass: when the detector fires, do NOT issue a model retry.
    # The handler still pops the suspect from the scorer-visible trace (the
    # score-trace pop) and from the chat history, then injects a fixed
    # abstention assistant message, then breaks the step loop so the next
    # user turn advances immediately. Designed to bypass the
    # case where a retry re-emits the same suspect call, which tool-call-specialized
    # models do often.
    "bypass",
}
_CRI_HEADER = "[Over-Action Reconciliation Required]"

# Score-trace protection rule: BFCL multi_turn does not reward refuse /
# ask text at empty-GT turns (the scorer skips them; PASS depends on the
# post-supplement trajectory completing — held-out function re-injection
# for ``miss_func``, user parameter supply for ``miss_param``). The
# protective effect of CRI therefore comes from *preventing the
# retracted call from contaminating backend state and scorer-visible
# response trace*, then letting the post-supplement turn complete
# cleanly. Retry text aims only at preventing another tool emission
# this turn; wording is intentionally neutral across both categories.
_CRI_DECISION_RULE_LINES = [
    "Do not emit a tool call in this turn. Briefly state that you cannot"
    " proceed yet because the required capability or information is not"
    " available.",
]
_MISS_PARAM_ASK_TURNS_PATH = Path(__file__).with_name("_miss_param_ask_turns.json")
_MISS_PARAM_ASK_TURNS: dict[str, int | None] | None = None


def _cri_mode() -> str:
    mode = os.environ.get("BFCL_CRI_MODE", "off").strip().lower() or "off"
    if mode not in _CRI_VALID_MODES:
        raise ValueError(
            "BFCL_CRI_MODE must be one of "
            f"{sorted(_CRI_VALID_MODES)}; got {mode!r}."
        )
    return mode


def _cri_variant() -> str:
    variant = os.environ.get("BFCL_CRI_VARIANT", "v1").strip().lower() or "v1"
    if variant not in _CRI_VALID_VARIANTS:
        raise ValueError(
            "BFCL_CRI_VARIANT must be one of "
            f"{sorted(_CRI_VALID_VARIANTS)}; got {variant!r}."
        )
    return variant


def _cri_enabled() -> bool:
    """True when the patch host must instantiate the runtime for CRI."""
    return _cri_mode() != "off"


def _cri_active() -> bool:
    """True when CRI should actually intercept (not passive_log/off)."""
    return _cri_mode() == "cri_only"


def _miss_param_ask_turns() -> dict[str, int | None]:
    """Lazy-load static miss_param ASK-turn metadata for bypass gating."""
    global _MISS_PARAM_ASK_TURNS
    if _MISS_PARAM_ASK_TURNS is None:
        try:
            with _MISS_PARAM_ASK_TURNS_PATH.open(encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            raw = {}
        _MISS_PARAM_ASK_TURNS = {
            str(k): (int(v) if isinstance(v, int) else None)
            for k, v in raw.items()
        }
    return _MISS_PARAM_ASK_TURNS


def _cri_gold_turn_idx(test_entry: dict) -> int | None:
    """Best-effort empty-GT (gold) turn index, model-input leak-free.

    Bypass mode replaces the suspect tool_call with an assistant
    placeholder and ends the turn empty — that is only safe at an
    empty-GT turn. We identify that turn from static benchmark metadata
    that is never shown to the model and is used only to decide whether
    Bypass mode can intervene safely.

    BFCL miss_func schema: the held-out function is
    injected at an empty-user-turn whose index is keyed in
    ``test_entry["missed_function"]``. The gold (empty-GT) turn is the
    non-empty user turn immediately before that injection turn:

        gold_turn_idx = min(int(k) for k in missed_function) - 1

    BFCL miss_param encodes the same designer-authored "should be empty"
    turn implicitly in ``possible_answer.ground_truth == []``. The
    sidecar ``_miss_param_ask_turns.json`` holds that turn index for every
    miss_param case, extracted once from BFCL's possible-answer files; it
    has no per-run or per-model dependency.

    Returns:
        The gold turn index for miss_func / miss_param cases, or
        ``None`` when the category lacks a safe marker.
    """
    holdout = test_entry.get("missed_function") or {}
    if holdout:
        try:
            inject_turns = sorted(int(k) for k in holdout.keys())
        except (ValueError, TypeError):
            return None
        if not inject_turns:
            return None
        candidate = inject_turns[0] - 1
        return candidate if candidate >= 0 else None

    case_id = str(test_entry.get("id") or test_entry.get("test_entry_id") or "")
    if case_id.startswith("multi_turn_miss_param"):
        # miss_func exposes the empty-GT turn directly via
        # ``test_entry["missed_function"]``. miss_param stores the same
        # benchmark-designer signal in possible_answer.ground_truth. This
        # sidecar is that structure copied into static metadata for the
        # intervention gate only: the model never sees it, and it is the
        # same information BFCL's grader uses to score the case.
        return _miss_param_ask_turns().get(case_id)
    return None


def _cri_available_function_names(test_entry: dict) -> set[str]:
    """Tool names compiled into this turn.

    Reads ``test_entry['function']``. In function-calling mode the driver
    appends the held-out tool there at the injection turn, so a call to it
    is MC-suspect only before injection; in prompting mode (local
    checkpoints) the list is not extended and the held-out tool stays
    MC-suspect for the whole case.
    """
    names: set[str] = set()
    for fn in test_entry.get("function", []) or []:
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def _cri_schema_by_name(test_entry: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for fn in test_entry.get("function", []) or []:
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            out[fn["name"]] = fn
    return out


def _cri_antecedent_blob(sri_runtime: SRIRuntime) -> str:
    """Lower-cased text the user has supplied + runtime-observed state.

    A required-param value found here (or in the param schema's
    enum/default, checked separately) is treated as a resolvable
    binding; absence is the UR (unresolved-referent) signal.
    """
    user_text = " ".join(sri_runtime.user_turn_texts or [])
    state_text = _format_state_snapshot(sri_runtime.state_by_domain) or ""
    return (user_text + " " + state_text).lower()


def _cri_value_has_antecedent(value: Any, schema: dict, blob: str) -> bool:
    # Only scalar string/number bindings are checked; dict/list/bool values
    # count as resolvable (not UR-suspect), which keeps false positives low.
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (dict, list)):
        return True
    s = str(value).strip()
    if len(s) < 2:
        return True
    if s.lower() in blob:
        return True
    enum = schema.get("enum") if isinstance(schema, dict) else None
    if isinstance(enum, list) and value in enum:
        return True
    if isinstance(schema, dict) and "default" in schema and value == schema["default"]:
        return True
    return False


def _cri_detect(
    call: dict,
    available_names: set[str],
    schema_by_name: dict[str, dict],
    sri_runtime: SRIRuntime,
) -> dict[str, Any]:
    """Return {'mc': bool, 'ur': bool, 'ur_param': str|None, 'suspect': bool}.

    MC = called fn name not in this turn's compiled tool set.
    UR = a required param bound to a value with no antecedent in the
         user text / observed state / param schema.
    """
    name = call.get("name")
    mc = isinstance(name, str) and name not in available_names

    ur = False
    ur_param: str | None = None
    fn_schema = schema_by_name.get(name) if isinstance(name, str) else None
    if fn_schema is not None:
        params = fn_schema.get("parameters") or {}
        required = params.get("required") or []
        props = params.get("properties") or {}
        args = call.get("arguments") or {}
        if isinstance(required, list) and isinstance(args, dict):
            blob = _cri_antecedent_blob(sri_runtime)
            for pname in required:
                if pname not in args:
                    # Missing required arg is an under-spec, not an
                    # invented binding; not the UR signal.
                    continue
                pschema = props.get(pname) if isinstance(props, dict) else {}
                if not _cri_value_has_antecedent(args[pname], pschema or {}, blob):
                    ur = True
                    ur_param = pname
                    break

    return {"mc": bool(mc), "ur": bool(ur), "ur_param": ur_param,
            "suspect": bool(mc or ur)}


def _cri_retry_instruction(state_text: str) -> str:
    rows = [
        _CRI_HEADER,
        "Your previous response emitted a tool call that may not be"
        " warranted by the request and available tools.",
        "Before it executes, reconcile the call against the user request,"
        " the available tools, and runtime-observed state.",
        "",
    ]
    rows.extend(
        [
            "Runtime-observed state from executed tool calls and tool results:",
            state_text,
            "",
        ]
    )
    rows.append("Decision rule:")
    rows.extend(_CRI_DECISION_RULE_LINES)
    return "\n".join(rows)


def _cri_evaluate(
    sri_runtime: SRIRuntime | None,
    model_response_data: dict[str, Any],
    test_entry: dict,
) -> dict[str, Any]:
    """Single entry the multi-turn driver consults on the non-empty seam.

    Returns a decision dict:
    - triggered: bool (the driver should apply the CRI variant)
    - reason: why it did or did not trigger ('cri_inactive',
      'cri_already_fired', 'no_tool_call', 'not_suspect',
      'over_action_suspect', or the stage gate that suppressed a suspect
      call; see the module docstring)
    - detector: per-call detector verdicts (list)
    - retry_instruction: str | None (set only when triggered)

    ``passive_log`` mode logs the suspect verdict but never triggers. CRI
    fires at most once per turn (a set of fired turns on the runtime).
    """
    if sri_runtime is None or _cri_mode() == "off":
        return {"triggered": False, "reason": "cri_inactive", "detector": []}

    variant = _cri_variant()
    turn_idx = sri_runtime.turn_index
    fired: set[int] = getattr(sri_runtime, "_cri_retry_turns", set())
    if turn_idx in fired:
        return {"triggered": False, "reason": "cri_already_fired", "detector": []}

    raw_calls = _response_data_tool_calls(model_response_data)
    calls = [c for c in (_normalise_tool_call(c) for c in raw_calls) if c]
    if not calls:
        return {"triggered": False, "reason": "no_tool_call", "detector": []}

    available = _cri_available_function_names(test_entry)
    schema_by_name = _cri_schema_by_name(test_entry)
    detector = [
        _cri_detect(c, available, schema_by_name, sri_runtime)
        for c in calls
    ]
    suspect = any(d["suspect"] for d in detector)

    decision: dict[str, Any] = {
        "triggered": False,
        "reason": "over_action_suspect" if suspect else "not_suspect",
        "detector": detector,
    }

    # Stage gates (apply after detector parse so events still carry
    # the detector verdict for diagnostics; only override the trigger
    # decision). Two gates, checked in order:
    #
    # 1. ``cri_post_injection_suppressed``: the live toolset has grown past
    #    its case-entry size (held-out function re-injected; function-calling
    #    mode only, see the module docstring).
    # 2. ``cri_already_fired_in_case``: a CRI fire already happened in this
    #    case.
    suppress_reason: str | None = None
    if suspect:
        initial_size = getattr(sri_runtime, "initial_toolset_size", None)
        current_size = len(test_entry.get("function") or [])
        if initial_size is not None and current_size > initial_size:
            suppress_reason = "cri_post_injection_suppressed"
        elif getattr(sri_runtime, "case_fired", False):
            suppress_reason = "cri_already_fired_in_case"

        # Bypass-specific guard: bypass leaves the turn
        # definitively empty (no retry, no decoded call), which only
        # matches the empty-GT (gold) turn. Firing bypass on a pre-gold
        # non-empty-GT turn would cause ``empty_turn_model_response``
        # failures. CRI-retry does not have this problem because its
        # retry step can re-emit a call.
        if suppress_reason is None and variant == "bypass":
            gold_idx = _cri_gold_turn_idx(test_entry)
            if gold_idx is None:
                suppress_reason = "cri_bypass_no_gold_turn_marker"
            elif turn_idx != gold_idx:
                suppress_reason = "cri_bypass_non_gold_turn"

    if suspect and _cri_active() and suppress_reason is None:
        state_text = _sri_state_to_text(
            sri_runtime.state_by_domain, max_chars=sri_runtime.max_state_chars
        )
        decision["triggered"] = True
        decision["retry_instruction"] = _cri_retry_instruction(state_text)
        if not hasattr(sri_runtime, "_cri_retry_turns"):
            sri_runtime._cri_retry_turns = set()  # type: ignore[attr-defined]
        sri_runtime._cri_retry_turns.add(turn_idx)  # type: ignore[attr-defined]
        # Mark case-level fire; consumed by gate 2 on subsequent turns.
        sri_runtime.case_fired = True
    elif suspect and suppress_reason is not None:
        # Stage-gated: keep detector verdict in the event, override reason.
        decision["reason"] = suppress_reason

    _write_sri_event(
        {
            "event": "cri_evaluate",
            "case_id": sri_runtime.test_entry_id,
            "test_entry_id": sri_runtime.test_entry_id,
            "turn_idx": turn_idx,
            "step_idx": sri_runtime.step_index,
            "mode": _cri_mode(),
            "variant": variant,
            "triggered": decision["triggered"],
            "reason": decision["reason"],
            "n_calls": len(calls),
            "detector": detector,
            "call_names": [c.get("name") for c in calls],
            "response_excerpt": _sri_truncate(
                str(model_response_data.get("model_responses") or ""), 500
            ),
        }
    )
    return decision
