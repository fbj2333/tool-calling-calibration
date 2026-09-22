from __future__ import annotations

import pytest

from bfcl_calibration.evaluation import bfcl
from bfcl_calibration.evaluation.bfcl._sri_loop import _handle_halt_candidate


def _render_llama_prompt(messages: list[dict[str, str]]) -> str:
    rendered = "<|begin_of_text|>"
    for message in messages:
        rendered += (
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content'].strip()}<|eot_id|>"
        )
    rendered += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return rendered


def test_sri_runtime_tracks_state_and_renders_retry_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    runtime = bfcl.SRIRuntime("multi_turn_base_11", "sri_only")
    runtime.start_turn([{"role": "user", "content": "Start the engine."}])
    runtime.observe_execution_results(
        [{"name": "startEngine", "arguments": {"ignitionMode": "START"}}],
        ['{"result": "engine started"}'],
    )
    runtime.start_turn([{"role": "user", "content": "Now fill the fuel tank."}])

    decision = runtime.observe_halt_candidate(
        {"model_responses": "All tasks are complete."},
        decode_error=None,
        is_empty_response=True,
    )
    prompt = runtime.retry_instruction()

    assert decision["retry"] is True
    assert prompt.startswith("[State Reconciliation Required]")
    assert "runtime-observed state" in prompt
    assert "vehicle.engine_state: running" in prompt
    assert "prior-1: Start the engine." in prompt
    assert "Now fill the fuel tank." in prompt
    assert "oracle" not in prompt.lower()
    assert "ground_truth" not in prompt


def test_toolace_sri_retry_survives_llama_prompt_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    toolace_system = (
        "ToolACE-2-8B prompt mode.\n"
        "Available tools:\n"
        '{"name":"lookup_order","parameters":{"type":"object"}}'
    )
    inference_data = {
        "message": [
            {"role": "system", "content": toolace_system},
            {"role": "user", "content": "Ship order 42."},
            {"role": "assistant", "content": "Done."},
        ]
    }
    runtime = bfcl.SRIRuntime("multi_turn_base_12", "sri_only", "v1")
    runtime.start_turn([{"role": "user", "content": "Ship order 42."}])
    runtime.observe_execution_results(
        [{"name": "lookup_order", "arguments": {"order_id": "42"}}],
        ['{"status": "ready"}'],
    )
    log: list[dict[str, object]] = []

    action = _handle_halt_candidate(
        sri_runtime=runtime,
        model_response_data={"model_responses": "Done."},
        inference_data=inference_data,
        current_step_inference_log=log,
        decode_error=None,
    )

    assert action == "continue"
    assert inference_data["message"][-1]["role"] == "system"
    retry_instruction = inference_data["message"][-1]["content"]
    assert "[State Reconciliation Required]" in retry_instruction
    assert "runtime-observed state" in retry_instruction
    for upstream_stop_token in (
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ):
        assert upstream_stop_token not in retry_instruction

    rendered = _render_llama_prompt(inference_data["message"])
    assert "ToolACE-2-8B prompt mode." in rendered
    assert '"name":"lookup_order"' in rendered
    assert retry_instruction in rendered
    assert "Done." not in rendered


def test_sri_runtime_retries_once_per_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    runtime = bfcl.SRIRuntime("multi_turn_base_12", "sri_only")
    runtime.start_turn([{"role": "user", "content": "Send the message."}])
    response = {"model_responses": "Done."}

    first = runtime.observe_halt_candidate(
        response,
        decode_error=None,
        is_empty_response=True,
    )
    second = runtime.observe_halt_candidate(
        response,
        decode_error=None,
        is_empty_response=True,
    )

    assert first["retry"] is True
    assert second["retry"] is False
    prompt = runtime.retry_instruction()
    assert not prompt.startswith("If the task is complete")
    assert "emit exactly one progress-making tool_call" in prompt





def test_sri_ablation_no_history_keeps_prompt_disables_history_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    runtime = bfcl.SRIRuntime(
        "multi_turn_base_12",
        "sri_only",
        "ablation_no_history",
    )
    runtime.start_turn([{"role": "user", "content": "Send the message."}])
    runtime.observe_execution_results(
        [{"name": "startEngine", "arguments": {"ignitionMode": "START"}}],
        ['{"result": "engine started"}'],
    )

    prompt = runtime.retry_instruction()

    # Prompt content unchanged from v1; full SRI body present.
    assert "[State Reconciliation Required]" in prompt
    assert "runtime-observed state" in prompt
    assert "vehicle.engine_state: running" in prompt
    assert "Decision rule:" in prompt
    # Toggle exposed for the retry call site to skip history-edit.
    assert runtime.disable_history_edit is True


def test_sri_ablation_no_state_omits_state_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    runtime = bfcl.SRIRuntime(
        "multi_turn_base_12",
        "sri_only",
        "ablation_no_state",
    )
    runtime.start_turn([{"role": "user", "content": "Send the message."}])
    runtime.observe_execution_results(
        [{"name": "startEngine", "arguments": {"ignitionMode": "START"}}],
        ['{"result": "engine started"}'],
    )

    prompt = runtime.retry_instruction()

    # Header + decision rule still present.
    assert "[State Reconciliation Required]" in prompt
    assert "Decision rule:" in prompt
    # State section removed.
    assert "Runtime-observed state from executed tool calls" not in prompt
    assert "vehicle.engine_state: running" not in prompt
    # History-edit still on (only state was ablated).
    assert runtime.disable_history_edit is False


def test_sri_dr_v1_should_skip_retry_only_for_dr_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    asking_data = {"model_responses": "Could you provide the order id?"}
    completion_data = {"model_responses": "Order placed successfully."}

    # decision_routed_v1: skips on asking, not on completion
    runtime_dr = bfcl.SRIRuntime(
        "multi_turn_miss_param_102",
        "sri_only",
        "decision_routed_v1",
    )
    assert runtime_dr.should_skip_retry(asking_data) is True
    assert runtime_dr.should_skip_retry(completion_data) is False

    # v1 (or other variants): never skips
    runtime_v1 = bfcl.SRIRuntime("multi_turn_miss_param_102", "sri_only", "v1")
    assert runtime_v1.should_skip_retry(asking_data) is False
    assert runtime_v1.should_skip_retry(completion_data) is False

    # ablation_no_state: never skips (different ablation, not DR)
    runtime_no_state = bfcl.SRIRuntime(
        "multi_turn_miss_param_102",
        "sri_only",
        "ablation_no_state",
    )
    assert runtime_no_state.should_skip_retry(asking_data) is False


def test_sri_dr_v1_variant_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    # Should not raise on construction
    runtime = bfcl.SRIRuntime(
        "multi_turn_miss_param_102",
        "sri_only",
        "decision_routed_v1",
    )
    runtime.start_turn([{"role": "user", "content": "Buy some Tesla shares at $700."}])
    # Variant produces standard SRI v1 retry text when retry is not skipped
    prompt = runtime.retry_instruction()
    assert "[State Reconciliation Required]" in prompt
    assert "Decision rule:" in prompt


def test_sri_dr_v2_a_fires_only_on_miss_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v2-A: detector should fire on miss_param case_ids and skip on others."""
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    asking_data = {"model_responses": "Could you provide the order id?"}

    # miss_param case_id: detector should fire
    runtime_mp = bfcl.SRIRuntime(
        "multi_turn_miss_param_102",
        "sri_only",
        "decision_routed_v2_a",
    )
    assert runtime_mp.should_skip_retry(asking_data) is True

    # base case_id: category gate blocks, detector should NOT fire
    runtime_base = bfcl.SRIRuntime(
        "multi_turn_base_117",
        "sri_only",
        "decision_routed_v2_a",
    )
    assert runtime_base.should_skip_retry(asking_data) is False

    # long_context case_id: category gate blocks
    runtime_lc = bfcl.SRIRuntime(
        "multi_turn_long_context_5",
        "sri_only",
        "decision_routed_v2_a",
    )
    assert runtime_lc.should_skip_retry(asking_data) is False

    # miss_func case_id: category gate blocks
    runtime_mf = bfcl.SRIRuntime(
        "multi_turn_miss_func_30",
        "sri_only",
        "decision_routed_v2_a",
    )
    assert runtime_mf.should_skip_retry(asking_data) is False


def test_sri_dr_v1_unchanged_by_v2_a_addition(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding v2-A must not affect v1 behavior on any case_id."""
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    asking_data = {"model_responses": "Could you provide the order id?"}

    # v1 should still fire on miss_param
    runtime_v1_mp = bfcl.SRIRuntime(
        "multi_turn_miss_param_102",
        "sri_only",
        "decision_routed_v1",
    )
    assert runtime_v1_mp.should_skip_retry(asking_data) is True

    # v1 has no category gate, so it also fires on base
    runtime_v1_base = bfcl.SRIRuntime(
        "multi_turn_base_117",
        "sri_only",
        "decision_routed_v1",
    )
    assert runtime_v1_base.should_skip_retry(asking_data) is True


def test_sri_ablation_no_recon_keeps_only_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BFCL_SRI_LOG_PATH", raising=False)
    runtime = bfcl.SRIRuntime(
        "multi_turn_base_12",
        "sri_only",
        "ablation_no_recon",
    )
    runtime.start_turn([{"role": "user", "content": "Send the message."}])
    runtime.observe_execution_results(
        [{"name": "startEngine", "arguments": {"ignitionMode": "START"}}],
        ['{"result": "engine started"}'],
    )

    prompt = runtime.retry_instruction()

    # State section only.
    assert prompt.startswith("Runtime-observed state from executed tool calls")
    assert "Runtime-observed state from executed tool calls" in prompt
    assert "vehicle.engine_state: running" in prompt
    # Reconciliation header / decision rule / prior turns removed.
    assert "[State Reconciliation Required]" not in prompt
    assert "Decision rule:" not in prompt
    assert "Prior user requests" not in prompt
    assert "Current user request" not in prompt
    # History-edit still on (only recon prompt was ablated).
    assert runtime.disable_history_edit is False
