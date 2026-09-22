"""Focused tests for CRI bypass gold-turn gate metadata."""
from __future__ import annotations

from bfcl_calibration.evaluation import bfcl
from bfcl_calibration.evaluation.bfcl import _cri


_FN = {
    "name": "post_tweet",
    "parameters": {
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
}


def _runtime(case_id: str = "multi_turn_miss_param_42", turn_index: int = 1):
    rt = bfcl.SRIRuntime(case_id, "passive_log", "v1")
    rt.turn_index = turn_index
    rt.step_index = 0
    rt.user_turn_texts = ["post something for me"]
    rt.state_by_domain = {}
    rt.initial_toolset_size = 1
    rt.case_fired = False
    return rt


def _resp():
    return {
        "model_responses": (
            '[{"name": "post_tweet", '
            '"arguments": {"content": "invented unseen text"}}]'
        )
    }


def test_miss_func_gold_turn_path_unchanged():
    entry = {
        "id": "multi_turn_miss_func_0",
        "function": [_FN],
        "missed_function": {
            "4": [{"name": "post_tweet", "parameters": {"type": "object"}}],
        },
    }
    assert _cri._cri_gold_turn_idx(entry) == 3


def test_miss_param_sidecar_hit_returns_ask_turn(monkeypatch):
    monkeypatch.setattr(
        _cri,
        "_MISS_PARAM_ASK_TURNS",
        {"multi_turn_miss_param_42": 2},
    )
    entry = {"id": "multi_turn_miss_param_42", "function": [_FN]}
    assert _cri._cri_gold_turn_idx(entry) == 2


def test_miss_param_sidecar_miss_suppresses_bypass(monkeypatch):
    monkeypatch.setattr(_cri, "_MISS_PARAM_ASK_TURNS", {})
    monkeypatch.setenv("BFCL_CRI_MODE", "cri_only")
    monkeypatch.setenv("BFCL_CRI_VARIANT", "bypass")
    entry = {"id": "multi_turn_miss_param_999", "function": [_FN]}
    decision = bfcl._cri_evaluate(_runtime("multi_turn_miss_param_999"), _resp(), entry)
    assert decision["triggered"] is False
    assert decision["reason"] == "cri_bypass_no_gold_turn_marker"
    assert decision["detector"][0]["suspect"] is True
