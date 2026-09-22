"""Unit tests for the CRI (Call-Reconciliation Intervention) probe.

Host-side and runnable without bfcl-eval installed: the detectors are
pure functions over (call, compiled tool list, SRIRuntime state); the
retry build + history edit are exercised with a real ``SRIRuntime``
constructed in ``passive_log`` mode (no vendor deps) plus a plain
message list.
"""
from __future__ import annotations

import pytest

from bfcl_calibration.evaluation import bfcl


def _runtime(**kw):
    rt = bfcl.SRIRuntime("multi_turn_miss_func_0", "passive_log", "v1")
    rt.turn_index = kw.get("turn_index", 2)
    rt.step_index = kw.get("step_index", 0)
    rt.user_turn_texts = kw.get("user_turn_texts", ["please sort the report"])
    rt.state_by_domain = kw.get("state_by_domain", {})
    return rt


_FN = {
    "name": "post_tweet",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tags": {"type": "string", "enum": ["#tech", "#news"]},
        },
        "required": ["content"],
    },
}
_TEST_ENTRY = {"id": "multi_turn_miss_func_0", "function": [_FN]}


# --- config-key validation -------------------------------------------------

def test_mode_variant_defaults_off(monkeypatch):
    monkeypatch.delenv("BFCL_CRI_MODE", raising=False)
    monkeypatch.delenv("BFCL_CRI_VARIANT", raising=False)
    assert bfcl._cri_mode() == "off"
    assert bfcl._cri_variant() == "v1"
    assert bfcl._cri_enabled() is False
    assert bfcl._cri_active() is False


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "bogus")
    with pytest.raises(ValueError, match="BFCL_CRI_MODE"):
        bfcl._cri_mode()


def test_invalid_variant_raises(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bogus")
    with pytest.raises(ValueError, match="BFCL_CRI_VARIANT"):
        bfcl._cri_variant()


# --- MC detector -----------------------------------------------------------

def test_mc_fires_when_called_fn_not_in_toolset():
    rt = _runtime()
    call = {"name": "delete_dir", "arguments": {}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    assert d["mc"] is True and d["suspect"] is True


def test_mc_silent_when_called_fn_present():
    rt = _runtime()
    call = {"name": "post_tweet", "arguments": {"content": "please sort the report"}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    assert d["mc"] is False


# --- UR detector -----------------------------------------------------------

def test_ur_fires_on_invented_required_binding():
    rt = _runtime(user_turn_texts=["post something for me"], state_by_domain={})
    call = {"name": "post_tweet", "arguments": {"content": "Quarterly revenue is 4.2M"}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    assert d["ur"] is True and d["ur_param"] == "content" and d["suspect"] is True


def test_ur_silent_when_value_has_user_antecedent():
    rt = _runtime(user_turn_texts=["tweet exactly: hello world"])
    call = {"name": "post_tweet", "arguments": {"content": "hello world"}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    assert d["ur"] is False


def test_ur_silent_when_value_in_state():
    rt = _runtime(user_turn_texts=["post it"], state_by_domain={"twitter": {"draft": "launch day"}})
    call = {"name": "post_tweet", "arguments": {"content": "launch day"}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    assert d["ur"] is False


def test_ur_silent_when_value_in_enum():
    rt = _runtime(user_turn_texts=["tag it appropriately"])
    call = {"name": "post_tweet", "arguments": {"content": "tag it appropriately", "tags": "#tech"}}
    d = bfcl._cri_detect(call, {"post_tweet"}, {"post_tweet": _FN}, rt)
    # content has antecedent (verbatim in user text); tags is enum-valid.
    assert d["ur"] is False


# --- variant detector masking ---------------------------------------------



# --- retry instruction builder (inverted-rule text verbatim) -------------

_INVERTED_RULE = "Do not emit a tool call in this turn."


def test_retry_v1_has_header_state_and_inverted_rule():
    msg = bfcl._cri_retry_instruction("STATE_SNAPSHOT")
    assert bfcl._CRI_HEADER in msg
    assert "STATE_SNAPSHOT" in msg
    assert _INVERTED_RULE in msg





# --- evaluate: mode gating + once-per-turn --------------------------------

def _resp(name="delete_dir"):
    return {"model_responses": f'[{{"name": "{name}", "arguments": {{}}}}]'}


def test_evaluate_off_never_triggers(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "off")
    rt = _runtime()
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is False and d["reason"] == "cri_inactive"


def test_evaluate_passive_log_detects_but_does_not_trigger(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "passive_log")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is False and d["reason"] == "over_action_suspect"


def test_evaluate_cri_only_triggers_once_per_turn(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    d1 = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d1["triggered"] is True
    assert d1["reason"] == "over_action_suspect"
    assert _INVERTED_RULE in d1["retry_instruction"]
    d2 = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d2["triggered"] is False and d2["reason"] == "cri_already_fired"


def test_evaluate_not_suspect_when_call_warranted(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    rt = _runtime(user_turn_texts=["please post: hello world"])
    resp = {"model_responses": '[{"name": "post_tweet", "arguments": {"content": "hello world"}}]'}
    d = bfcl._cri_evaluate(rt, resp, _TEST_ENTRY)
    assert d["triggered"] is False and d["reason"] == "not_suspect"


# --- driver hook: history edit + continue contract ------------------------

def test_handle_overaction_triggers_pop_and_append(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    inference_data = {"message": [{"role": "user", "content": "hi"},
                                  {"role": "assistant", "content": "[delete_dir()]"}]}
    log: list = []
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=log,
    )
    assert action == "continue"
    roles = [m["role"] for m in inference_data["message"]]
    assert roles == ["user", "system"]  # assistant popped, system retry appended
    assert _INVERTED_RULE in inference_data["message"][-1]["content"]
    assert log and log[0]["role"] == "handler_log"



def test_handle_overaction_not_triggered_returns_none(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "off")
    rt = _runtime()
    inference_data = {"message": [{"role": "assistant", "content": "x"}]}
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=[],
    )
    assert action is None
    assert [m["role"] for m in inference_data["message"]] == ["assistant"]


# --- score-trace protection -----------------------------------------------
#
# Without these pops the BFCL multi_turn scorer re-decodes the retracted
# raw output from ``all_model_response`` and re-executes it as if the
# runtime retract had never happened. These tests pin the
# rollback contract so the leak cannot regress silently.


def _per_step_buffers():
    """Mock what ``_inference_multi_turn`` appended to the per-step
    buffers just before consulting CRI (line 398–404 of ``_sri_loop``).
    """
    return {
        "current_turn_response": ["[delete_dir()]"],
        "current_turn_input_token_count": [123.0],
        "current_turn_output_token_count": [45.0],
        "current_turn_latency": [0.5],
        "current_turn_reasoning_content": [""],
    }


def test_handle_overaction_pops_score_trace_buffers(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    inference_data = {
        "message": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[delete_dir()]"},
        ]
    }
    buffers = _per_step_buffers()
    pre_lens = {k: len(v) for k, v in buffers.items()}
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=[],
        **buffers,
    )
    assert action == "continue"
    # All five per-step buffers must have been popped exactly once.
    for k, v in buffers.items():
        assert len(v) == pre_lens[k] - 1, f"{k} not popped"
    # The retracted raw output must no longer be in current_turn_response —
    # this is the property the BFCL scorer reads via all_model_response.
    assert "[delete_dir()]" not in buffers["current_turn_response"]


def test_handle_overaction_not_triggered_does_not_pop_buffers(monkeypatch):
    monkeypatch.setenv("BFCL_CRI_MODE", "passive_log")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    inference_data = {"message": [{"role": "assistant", "content": "x"}]}
    buffers = _per_step_buffers()
    pre_lens = {k: len(v) for k, v in buffers.items()}
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=[],
        **buffers,
    )
    # passive_log never returns 'continue'; the per-step buffers must
    # stay untouched so the scorer still sees the raw output.
    assert action is None
    for k, v in buffers.items():
        assert len(v) == pre_lens[k], f"{k} was modified despite no trigger"


def test_handle_overaction_without_buffer_args(monkeypatch):
    """Called without the five buffer args, only inference_data and the
    handler log are edited."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    inference_data = {
        "message": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[delete_dir()]"},
        ]
    }
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=[],
    )
    assert action == "continue"
    roles = [m["role"] for m in inference_data["message"]]
    assert roles == ["user", "system"]


# --- stage-aware suppression (post-injection + once-per-case) -------------
#
# Two SRIRuntime attributes drive these gates:
#   - ``initial_toolset_size`` (int | None): set by ``_inference_multi_turn``
#     at case entry; ``_cri_evaluate`` suppresses when the current toolset
#     exceeds it (held-out function re-injected; function-calling mode).
#   - ``case_fired`` (bool): flipped True once any CRI trigger fires;
#     subsequent fires in the same case are suppressed.


def test_post_injection_suppress_when_toolset_grew(monkeypatch):
    """If the live toolset has grown past initial_toolset_size, CRI must
    NOT trigger — the held-out tool has been re-injected and any tool
    call there is part of the legitimate post-injection trajectory."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    rt.initial_toolset_size = 1  # case entered with 1 function in toolset
    rt.case_fired = False
    grown_entry = {
        "id": "multi_turn_miss_func_0",
        # 2 functions now → toolset has grown past initial size. Neither
        # is ``delete_dir`` (the suspect call name in ``_resp()``) so MC
        # still fires; the stage gate has to be the thing that suppresses.
        "function": [
            _FN,
            {"name": "some_other_fn", "parameters": {"type": "object"}},
        ],
    }
    d = bfcl._cri_evaluate(rt, _resp(), grown_entry)
    assert d["triggered"] is False
    assert d["reason"] == "cri_post_injection_suppressed"
    # Detector verdict must still be carried in the event for diagnostics:
    # the suspect call (``delete_dir``) is NOT in the (grown) toolset, so
    # MC must fire even though the trigger is gate-suppressed.
    assert d["detector"], "detector verdict must be present"
    assert d["detector"][0]["suspect"] is True
    assert d["detector"][0]["mc"] is True


def test_post_injection_does_not_suppress_when_toolset_unchanged(monkeypatch):
    """If the toolset has not grown (still in the pre-injection phase),
    the existing trigger logic must run as before."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    rt.initial_toolset_size = len(_TEST_ENTRY["function"])  # unchanged
    rt.case_fired = False
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is True
    assert d["reason"] == "over_action_suspect"


def test_case_fired_suppresses_subsequent_fire(monkeypatch):
    """Once a fire has happened in the case (case_fired = True), no
    further fire is allowed — even on a later turn. This is the
    miss_param fallback (no toolset-grow signal there)."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime(turn_index=5)  # different turn from the (implicit) first
    rt.initial_toolset_size = len(_TEST_ENTRY["function"])
    rt.case_fired = True  # already fired earlier in this case
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is False
    assert d["reason"] == "cri_already_fired_in_case"


def test_first_fire_flips_case_fired(monkeypatch):
    """The first fire in a case must set case_fired = True so the
    once-per-case gate can suppress the next fire."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    rt.initial_toolset_size = len(_TEST_ENTRY["function"])
    rt.case_fired = False
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is True
    assert rt.case_fired is True


def test_runtime_without_suppression_attrs_still_triggers(monkeypatch):
    """A runtime lacking ``initial_toolset_size`` / ``case_fired`` falls back
    to ``getattr`` defaults (None / False), i.e. no suppression."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    # Drop the suppression attributes.
    if hasattr(rt, "initial_toolset_size"):
        delattr(rt, "initial_toolset_size")
    if hasattr(rt, "case_fired"):
        delattr(rt, "case_fired")
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is True
    assert d["reason"] == "over_action_suspect"


# --- Bypass mode variant (`bypass`) ---------------------------------------
#
# Bypass mode replaces the retry-instruction injection with a placeholder
# abstain message + a "break" return signal: the driver exits the step
# loop without retrying the decode. Designed for tool-call-specialized models where
# retry deterministically re-emits the same suspect tool call.


def test_bypass_variant_is_valid():
    """``bypass`` must be registered as a valid CRI_VARIANT value."""
    import os
    os.environ["BFCL_CRI_VARIANT"] = "bypass"
    try:
        assert bfcl._cri_variant() == "bypass"
    finally:
        os.environ.pop("BFCL_CRI_VARIANT", None)


def test_bypass_handle_returns_break_and_inserts_placeholder(monkeypatch):
    """Bypass mode: handler should return 'break' and insert an
    assistant abstain message in inference_data['message'] instead of a
    role:system retry instruction. The test_entry has a missed_function
    marker so the gold-turn guard passes (default _runtime turn_index=2
    matches gold_turn_idx=2 in _miss_func_entry)."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime()  # turn_index defaults to 2
    inference_data = {
        "message": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[delete_dir()]"},
        ]
    }
    buffers = _per_step_buffers()
    # Note: the _miss_func_entry helper is defined further down in this
    # file; we inline-construct an equivalent here to keep the test
    # ordering forward-compatible.
    miss_func_entry = {
        "id": "multi_turn_miss_func_test",
        "function": [_FN],
        "missed_function": {
            "3": [{"name": "delete_dir", "parameters": {"type": "object"}}],
        },
    }
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=miss_func_entry,
        current_step_inference_log=[],
        **buffers,
    )
    assert action == "break"
    roles = [m["role"] for m in inference_data["message"]]
    # The assistant suspect message was popped; a fresh assistant
    # placeholder replaces it. No role:system retry instruction is
    # appended (that is the retry-mode behavior).
    assert roles == ["user", "assistant"]
    last_msg = inference_data["message"][-1]["content"]
    assert "cannot proceed" in last_msg.lower()
    # Score-trace pops still happen.
    for v in buffers.values():
        assert len(v) == 0


def test_v1_variant_unchanged_after_bypass_added(monkeypatch):
    """Adding ``bypass`` must not alter v1's retry behavior."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime()
    inference_data = {
        "message": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[delete_dir()]"},
        ]
    }
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_TEST_ENTRY,
        current_step_inference_log=[],
    )
    assert action == "continue"
    roles = [m["role"] for m in inference_data["message"]]
    assert roles == ["user", "system"]  # v1 still appends role:system


# --- bypass gold-turn guard -----------------------------------------------
#
# Bypass leaves the turn definitively empty (no retry, no decoded call).
# That is only safe at the empty-GT (gold) turn. On pre-gold non-empty-GT
# turns BFCL scorer would mark the turn ``empty_turn_model_response``.
# A separate stage gate suppresses bypass when the runtime cannot
# identify the gold turn (miss_param / base) OR when the current turn
# is not the gold turn (pre-gold fire).


def _miss_func_entry(gold_turn_idx: int):
    """Build a test_entry mimicking miss_func: missed_function keyed by
    the holdout-injection turn (gold_turn_idx + 1)."""
    return {
        "id": "multi_turn_miss_func_test",
        "function": [_FN],
        "missed_function": {
            str(gold_turn_idx + 1): [
                {"name": "delete_dir", "parameters": {"type": "object"}},
            ],
        },
    }


def test_bypass_fires_at_gold_turn_when_marker_present(monkeypatch):
    """Bypass mode at the empty-GT (gold) turn must fire normally."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime(turn_index=2)
    rt.initial_toolset_size = 1
    rt.case_fired = False
    entry = _miss_func_entry(gold_turn_idx=2)
    d = bfcl._cri_evaluate(rt, _resp(), entry)
    assert d["triggered"] is True
    assert d["reason"] == "over_action_suspect"


def test_bypass_suppressed_on_pre_gold_non_empty_gt_turn(monkeypatch):
    """Bypass mode at a pre-gold (non-empty-GT) turn must NOT fire —
    that would empty a turn where BFCL expects a tool call."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime(turn_index=0)  # pre-gold (gold is at idx 2)
    rt.initial_toolset_size = 1
    rt.case_fired = False
    entry = _miss_func_entry(gold_turn_idx=2)
    d = bfcl._cri_evaluate(rt, _resp(), entry)
    assert d["triggered"] is False
    assert d["reason"] == "cri_bypass_non_gold_turn"
    # Detector still records the suspect verdict for diagnostics.
    assert d["detector"][0]["suspect"] is True


def test_bypass_suppressed_without_gold_turn_marker(monkeypatch):
    """Bypass mode on miss_param / base (no ``missed_function``) cannot
    identify a gold turn at runtime, so it must suppress to avoid
    accidentally emptying non-empty-GT turns. v1 retry mode is
    unaffected by this gate."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime()
    rt.initial_toolset_size = 1
    rt.case_fired = False
    # _TEST_ENTRY has no missed_function field — miss_param-like case.
    d = bfcl._cri_evaluate(rt, _resp(), _TEST_ENTRY)
    assert d["triggered"] is False
    assert d["reason"] == "cri_bypass_no_gold_turn_marker"


def test_v1_unaffected_by_bypass_guards(monkeypatch):
    """The Bypass-Mode-specific gates apply only to ``bypass``; v1
    retry must still fire on a pre-gold turn or in the absence of a
    gold-turn marker."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "v1")
    rt = _runtime(turn_index=0)
    rt.initial_toolset_size = 1
    rt.case_fired = False
    entry = _miss_func_entry(gold_turn_idx=2)  # gold elsewhere, but v1 fires
    d = bfcl._cri_evaluate(rt, _resp(), entry)
    assert d["triggered"] is True


def test_bypass_gate_priority_post_injection_wins(monkeypatch):
    """When a bypass case is both post-injection AND on a non-gold turn,
    the more informative diagnostic ``cri_post_injection_suppressed``
    must win over the bypass-only ``cri_bypass_non_gold_turn``. This
    pins the gate-order contract (post-injection > case_fired >
    bypass-specific)."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime(turn_index=5)  # not the gold turn (gold = 2)
    rt.initial_toolset_size = 1
    rt.case_fired = False
    # Toolset has grown → post-injection true; but turn_idx != gold,
    # so bypass guard would also fire. Test: post-injection reason wins.
    grown_entry = {
        "id": "multi_turn_miss_func_test",
        "function": [
            _FN,
            {"name": "some_other_fn", "parameters": {"type": "object"}},
        ],
        "missed_function": {
            "3": [{"name": "delete_dir", "parameters": {"type": "object"}}],
        },
    }
    d = bfcl._cri_evaluate(rt, _resp(), grown_entry)
    assert d["triggered"] is False
    assert d["reason"] == "cri_post_injection_suppressed"


def test_bypass_placeholder_not_in_scorer_buffer(monkeypatch):
    """The bypass placeholder lives in inference_data['message'] (so
    the next model turn sees coherent chat history) but must NOT enter
    current_turn_response (so the scorer sees an empty turn, matching
    empty-GT semantics)."""
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    rt = _runtime(turn_index=2)
    rt.initial_toolset_size = 1
    rt.case_fired = False
    inference_data = {
        "message": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "[delete_dir()]"},
        ]
    }
    buffers = _per_step_buffers()
    action = bfcl._handle_overaction_candidate(
        sri_runtime=rt,
        model_response_data=_resp(),
        inference_data=inference_data,
        test_entry=_miss_func_entry(gold_turn_idx=2),
        current_step_inference_log=[],
        **buffers,
    )
    assert action == "break"
    # Placeholder reaches the next model decode via inference_data.
    last_msg = inference_data["message"][-1]
    assert last_msg["role"] == "assistant"
    assert "cannot proceed" in last_msg["content"].lower()
    # But the scorer-visible buffer is empty (placeholder is NOT added
    # there; the only thing that happens is the pop of the suspect raw).
    assert buffers["current_turn_response"] == []
    assert "cannot proceed" not in str(buffers["current_turn_response"]).lower()
