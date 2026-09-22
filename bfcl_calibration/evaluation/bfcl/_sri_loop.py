"""Multi-turn driver + ``BaseHandler`` patch host for SRI and CRI.

``_inference_multi_turn`` is the wrapper-side replacement for
``BaseHandler.inference_multi_turn_{FC,prompting}``: it runs the same
turn / step loop the vendor handler does, but consults the active
``SRIRuntime`` (set per-test-entry via ``_apply_sri_patches``) at
halt-candidate boundaries to decide whether to inject an SRI
retry instruction or fall through to the next turn. CRI (the call-seam probe)
is consulted on the complement seam — the ``is_empty == False`` path
(>=1 parseable tool_call) — before ``execute_multi_turn_func_call``,
via ``_handle_overaction_candidate``.

``_apply_sri_patches`` installs the wrapper into ``BaseHandler`` once
``_load_bfcl_handles`` imports the upstream module. The patch is
gated by SRI active *or* CRI enabled
(``_sri_mode() != "off" or _cri_enabled()``) and idempotent
(``_bfclcalib_sri_patched`` sentinel attr). Non-multi-turn entries
go straight to the original vendor inference path.

The helpers ``_DEFAULT_RETRY_INSTRUCTION`` / ``_remove_last_assistant`` /
``_append_retry_instruction`` are used only by the loop body.
"""
from __future__ import annotations

from typing import Any

from ._sri import (
    _SRI_CONTEXT,
    SRIRuntime,
    _current_sri_runtime,
    _sri_mode,
    _sri_variant,
)
from ._state import _response_data_tool_calls
from ._cri import _cri_enabled, _cri_evaluate, _cri_variant


# Default fallback retry instruction used when the SRI runtime is asked to
# retry but does not supply its own message. SRI's standard retry_instruction
# always produces a non-empty string in practice; this constant is a safety
# net for the edge case.
_DEFAULT_RETRY_INSTRUCTION = (
    "If the task is complete, provide the final user-facing answer directly. "
    "Otherwise emit the next tool_call now."
)


def _remove_last_assistant(inference_data: dict[str, Any]) -> None:
    messages = inference_data.get("message")
    if not isinstance(messages, list) or not messages:
        return
    # Closed-API handlers (OpenAICompletionsHandler etc.) append a
    # pydantic ``ChatCompletionMessage`` object to ``message``; open-source
    # handlers append plain dicts. Support both. ``getattr`` on a dict
    # falls through to the dict-key path because dicts have no ``role``
    # attribute, so the order below is deliberate.
    last = messages[-1]
    if isinstance(last, dict):
        role = last.get("role")
    else:
        role = getattr(last, "role", None)
    if role == "assistant":
        messages.pop()


def _append_retry_instruction(inference_data: dict[str, Any], instruction: str) -> None:
    messages = inference_data.setdefault("message", [])
    messages.append({"role": "system", "content": instruction})


def _handle_halt_candidate(
    *,
    sri_runtime: SRIRuntime | None,
    model_response_data: dict[str, Any],
    inference_data: dict[str, Any],
    current_step_inference_log: list[dict[str, Any]],
    decode_error: str | None,
    decoded_model_responses: Any = None,
) -> str:
    """Run SRI checks on a halt candidate. Return 'continue' / 'break'.

    'continue' means the caller should ``count += 1; continue`` (a retry was
    injected); 'break' means the caller should exit the inner step-loop.
    """

    sri_decision = (
        sri_runtime.observe_halt_candidate(
            model_response_data,
            decode_error=decode_error,
            is_empty_response=True,
        )
        if sri_runtime is not None
        else {"retry": False}
    )
    if sri_decision.get("retry"):
        if sri_runtime is not None and sri_runtime.should_skip_retry(model_response_data):
            log = {
                "role": "handler_log",
                "content": (
                    "Decision-routed SRI: ask-user response detected, retry suppressed (decode-failure path)."
                    if decode_error
                    else "Decision-routed SRI: ask-user response detected, retry suppressed."
                ),
                "sri_decision": sri_decision,
            }
            if decode_error is not None:
                log["error"] = decode_error
            current_step_inference_log.append(log)
            return "break"
        if not (sri_runtime is not None and sri_runtime.disable_history_edit):
            _remove_last_assistant(inference_data)
        retry_instruction = None
        if sri_runtime is not None and sri_decision.get("retry"):
            retry_instruction = sri_runtime.retry_instruction()
        _append_retry_instruction(
            inference_data,
            retry_instruction or _DEFAULT_RETRY_INSTRUCTION,
        )
        log = {
            "role": "handler_log",
            "content": (
                "SRI retry injected after decode failure/no parseable tool call."
                if decode_error
                else "SRI retry injected after empty/no-call response."
            ),
            "sri_decision": sri_decision,
            "history_edit_skipped": bool(
                sri_runtime is not None and sri_runtime.disable_history_edit
            ),
        }
        if decode_error is not None:
            log["error"] = decode_error
        current_step_inference_log.append(log)
        return "continue"

    if decode_error is not None:
        print("Failed to decode the model response. Proceed to next turn.")
        current_step_inference_log.append(
            {
                "role": "handler_log",
                "content": "Error decoding the model response. Proceed to next turn.",
                "error": decode_error,
                "sri_decision": sri_decision,
            }
        )
    else:
        print("Empty response from the model. Proceed to next turn.")
        current_step_inference_log.append(
            {
                "role": "handler_log",
                "content": "Empty response from the model. Proceed to next turn.",
                "model_response_decoded": decoded_model_responses,
                "sri_decision": sri_decision,
            }
        )
    return "break"


def _handle_overaction_candidate(
    *,
    sri_runtime: SRIRuntime | None,
    model_response_data: dict[str, Any],
    inference_data: dict[str, Any],
    test_entry: dict,
    current_step_inference_log: list[dict[str, Any]],
    current_turn_response: list[Any] | None = None,
    current_turn_input_token_count: list[float] | None = None,
    current_turn_output_token_count: list[float] | None = None,
    current_turn_latency: list[float] | None = None,
    current_turn_reasoning_content: list[str] | None = None,
) -> str | None:
    """Consult CRI on the non-empty (>=1 parseable tool_call) seam.

    Mirror of ``_handle_halt_candidate`` for the complement branch, run
    *before* ``execute_multi_turn_func_call`` so a retracted call never
    mutates backend state. Returns 'continue' when an CRI retry was
    injected (caller does ``count += 1; continue``); None when CRI did
    not fire (caller proceeds to execute the call normally).

    When CRI fires the per-step buffers — ``current_turn_response`` plus
    its companion token / latency / reasoning lists — are popped in
    addition to the chat-history edit. This closes the BFCL scorer-leak:
    without the pop, the retracted raw output stays in
    ``all_model_response`` and the scorer (which re-decodes raw
    responses end-to-end) re-executes the suspect call as if the
    runtime retract had never happened, undoing the intervention and
    typically producing duplicate-call trajectory artifacts. The 5
    buffer arguments are optional (unit tests omit them); the driver in
    ``_inference_multi_turn`` passes all five.
    """
    decision = _cri_evaluate(sri_runtime, model_response_data, test_entry)
    if not decision.get("triggered"):
        return None

    variant = _cri_variant()

    # Score-trace protection: always pop the per-step buffers so
    # the scorer cannot re-decode the retracted raw output. Applies to
    # both retry-mode and bypass-mode variants — both want the suspect
    # call gone from the scorer-visible trajectory.
    _remove_last_assistant(inference_data)
    for buf in (
        current_turn_response,
        current_turn_input_token_count,
        current_turn_output_token_count,
        current_turn_latency,
        current_turn_reasoning_content,
    ):
        if buf is not None and buf:
            buf.pop()

    if variant == "bypass":
        # Bypass mode: do not retry the decode. Instead, insert a fixed
        # abstention assistant message into the chat history (so the next
        # user-turn is well-formed) and break out of the step loop. The
        # gold turn ends up empty in the scorer-visible trajectory, which
        # is exactly what miss_func / miss_param empty-GT turns expect.
        # The caller in ``_inference_multi_turn`` detects the "break"
        # return value and exits the inner step loop without executing
        # any tool call, advancing directly to the next turn.
        inference_data.setdefault("message", []).append(
            {
                "role": "assistant",
                "content": (
                    "I cannot proceed with this step yet; the required "
                    "capability or information is not available."
                ),
            }
        )
        current_step_inference_log.append(
            {
                "role": "handler_log",
                "content": (
                    "CRI bypass mode: over-action-suspect tool call "
                    "retracted pre-execute; no retry, placeholder abstain "
                    "inserted; scorer trace rolled back."
                ),
                "cri_decision": {
                    k: v for k, v in decision.items() if k != "retry_instruction"
                },
            }
        )
        return "break"

    # CRI-retry (v1).
    _append_retry_instruction(inference_data, decision["retry_instruction"])
    current_step_inference_log.append(
        {
            "role": "handler_log",
            "content": (
                "CRI retry injected: over-action-suspect tool call "
                "retracted pre-execute; per-step scorer trace also "
                "rolled back."
            ),
            "cri_decision": {
                k: v for k, v in decision.items() if k != "retry_instruction"
            },
        }
    )
    return "continue"


def _inference_multi_turn(
    self,
    test_entry: dict,
    include_input_log: bool,
    exclude_state_log: bool,
    *,
    base_handler_module: Any,
    fc_mode: bool,
) -> tuple[list[list], dict]:
    initial_config: dict = test_entry.get("initial_config", {})
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    test_category: str = test_entry_id.rsplit("_", 1)[0]
    holdout_function: dict[int, list] = test_entry.get("missed_function", {})

    total_input_token_count: list[list[float]] = []
    total_output_token_count: list[list[float]] = []
    total_latency: list[list[float]] = []
    all_model_response: list[list] = []
    all_inference_log: list[list[dict]] = []
    all_reasoning_content: list[list] = []
    force_quit = False

    _, involved_instances = base_handler_module.execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        self.model_name_underline_replaced,
        test_entry_id,
        long_context=("long_context" in test_category or "composite" in test_category),
        is_evaL_run=False,
    )

    if base_handler_module.is_memory(test_category):
        assert len(involved_instances) == 1, "Memory category should only involve one class."
        memory_instance = list(involved_instances.values())[0]
        test_entry["question"] = base_handler_module.add_memory_instruction_system_prompt(
            test_entry["question"],
            test_category,
            test_entry["scenario"],
            memory_instance,
        )

    if not exclude_state_log:
        state_log = []
        for class_name, class_instance in involved_instances.items():
            if (
                class_name in base_handler_module.STATELESS_CLASSES
                or class_name in base_handler_module.OMIT_STATE_INFO_CLASSES
            ):
                continue
            class_instance = base_handler_module.deepcopy(class_instance)
            state_log.append(
                {
                    "role": "state_info",
                    "class_name": class_name,
                    "content": {
                        key: value
                        for key, value in vars(class_instance).items()
                        if not key.startswith("_")
                    },
                }
            )
        if state_log:
            all_inference_log.append(state_log)

    if fc_mode:
        inference_data: dict = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
    else:
        inference_data = self._pre_query_processing_prompting(test_entry)

    sri_runtime = _current_sri_runtime()
    if sri_runtime is not None:
        # CRI: record the entry-time toolset size and reset the case-level
        # fire flag. In function-calling mode the holdout-injection turn below
        # extends ``test_entry["function"]``, which ``_cri_evaluate`` reads as
        # the post-injection signal; the prompting branch announces the
        # held-out function in the prompt only, so the list does not grow.
        sri_runtime.initial_toolset_size = len(test_entry.get("function") or [])
        sri_runtime.case_fired = False
    all_multi_turn_messages: list[list[dict]] = test_entry["question"]
    for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
        if str(turn_idx) in holdout_function:
            if fc_mode:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                inference_data = self._compile_tools(inference_data, test_entry)
                prompt = base_handler_module.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
            else:
                prompt = base_handler_module.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=holdout_function[str(turn_idx)]
                )
            assert len(current_turn_message) == 0, "Holdout turn should not have user message."
            current_turn_message = [{"role": "user", "content": prompt}]

        if sri_runtime is not None:
            sri_runtime.start_turn(current_turn_message)

        if turn_idx == 0:
            if fc_mode:
                inference_data = self.add_first_turn_message_FC(inference_data, current_turn_message)
            else:
                inference_data = self.add_first_turn_message_prompting(inference_data, current_turn_message)
        else:
            if fc_mode:
                inference_data = self._add_next_turn_user_message_FC(inference_data, current_turn_message)
            else:
                inference_data = self._add_next_turn_user_message_prompting(inference_data, current_turn_message)

        current_turn_response = []
        current_turn_inference_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}
        current_turn_input_token_count: list[float] = []
        current_turn_output_token_count: list[float] = []
        current_turn_latency: list[float] = []
        current_turn_reasoning_content: list[str] = []

        count = 0
        while True:
            print("-" * 100)
            print(f"ID: {test_entry_id.replace('multi_turn_', '')}, Turn: {turn_idx}, Step: {count}")
            current_step_inference_log: list[dict] = []
            current_turn_inference_log[f"step_{count}"] = current_step_inference_log

            if fc_mode:
                api_response, query_latency = self._query_FC(inference_data)
            else:
                api_response, query_latency = self._query_prompting(inference_data)

            if include_input_log:
                current_step_inference_log.append(
                    {
                        "role": "inference_input",
                        "content": inference_data.get("inference_input_log", ""),
                    }
                )

            if fc_mode:
                model_response_data = self._parse_query_response_FC(api_response)
            else:
                model_response_data = self._parse_query_response_prompting(api_response)
            model_responses = model_response_data["model_responses"]

            if fc_mode:
                inference_data = self._add_assistant_message_FC(inference_data, model_response_data)
            else:
                inference_data = self._add_assistant_message_prompting(inference_data, model_response_data)

            current_turn_input_token_count.append(model_response_data["input_token"])
            current_turn_output_token_count.append(model_response_data["output_token"])
            current_turn_latency.append(query_latency)
            current_turn_response.append(model_responses)

            reasoning_content = model_response_data.get("reasoning_content", "")
            current_turn_reasoning_content.append(reasoning_content)
            log_entry = {"role": "assistant", "content": model_responses}
            if reasoning_content:
                log_entry["reasoning_content"] = reasoning_content
            current_step_inference_log.append(log_entry)

            try:
                decoded_model_responses = self.decode_execute(model_responses, has_tool_call_tag=False)
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": "Successfully decoded model response.",
                        "model_response_decoded": decoded_model_responses,
                    }
                )
                is_empty = base_handler_module.is_empty_execute_response(decoded_model_responses)
                model_response_data["model_responses_decoded"] = decoded_model_responses
                if is_empty:
                    action = _handle_halt_candidate(
                        sri_runtime=sri_runtime,
                        model_response_data=model_response_data,
                        inference_data=inference_data,
                        current_step_inference_log=current_step_inference_log,
                        decode_error=None,
                        decoded_model_responses=decoded_model_responses,
                    )
                    if action == "continue":
                        count += 1
                        continue
                    break
                else:
                    cri_action = _handle_overaction_candidate(
                        sri_runtime=sri_runtime,
                        model_response_data=model_response_data,
                        inference_data=inference_data,
                        test_entry=test_entry,
                        current_step_inference_log=current_step_inference_log,
                        current_turn_response=current_turn_response,
                        current_turn_input_token_count=current_turn_input_token_count,
                        current_turn_output_token_count=current_turn_output_token_count,
                        current_turn_latency=current_turn_latency,
                        current_turn_reasoning_content=current_turn_reasoning_content,
                    )
                    if cri_action == "continue":
                        count += 1
                        continue
                    if cri_action == "break":
                        # Bypass mode: CRI retracted the suspect call,
                        # inserted a placeholder abstain, and asked the
                        # driver to skip execution + retry. Exit the step
                        # loop so the outer turn loop advances directly
                        # to the next user turn.
                        break
            except Exception as exc:
                action = _handle_halt_candidate(
                    sri_runtime=sri_runtime,
                    model_response_data=model_response_data,
                    inference_data=inference_data,
                    current_step_inference_log=current_step_inference_log,
                    decode_error=str(exc),
                )
                if action == "continue":
                    count += 1
                    continue
                break

            execution_results, involved_instances = base_handler_module.execute_multi_turn_func_call(
                decoded_model_responses,
                initial_config,
                involved_classes,
                self.model_name_underline_replaced,
                test_entry_id,
                long_context=("long_context" in test_category or "composite" in test_category),
                is_evaL_run=False,
            )
            if sri_runtime is not None:
                sri_runtime.observe_execution_results(
                    _response_data_tool_calls(model_response_data),
                    execution_results,
                )

            if fc_mode:
                inference_data = self._add_execution_results_FC(
                    inference_data, execution_results, model_response_data
                )
            else:
                inference_data = self._add_execution_results_prompting(
                    inference_data, execution_results, model_response_data
                )

            for execution_result in execution_results:
                current_step_inference_log.append({"role": "tool", "content": execution_result})

            count += 1
            if count > base_handler_module.MAXIMUM_STEP_LIMIT:
                force_quit = True
                current_step_inference_log.append(
                    {
                        "role": "handler_log",
                        "content": f"Model has been forced to quit after {base_handler_module.MAXIMUM_STEP_LIMIT} steps.",
                    }
                )
                break

        all_model_response.append(current_turn_response)
        all_inference_log.append(current_turn_inference_log)
        all_reasoning_content.append(current_turn_reasoning_content)
        total_input_token_count.append(current_turn_input_token_count)
        total_output_token_count.append(current_turn_output_token_count)
        total_latency.append(current_turn_latency)

        if not exclude_state_log:
            state_log = []
            for class_name, class_instance in involved_instances.items():
                if (
                    class_name in base_handler_module.STATELESS_CLASSES
                    or class_name in base_handler_module.OMIT_STATE_INFO_CLASSES
                ):
                    continue
                class_instance = base_handler_module.deepcopy(class_instance)
                state_log.append(
                    {
                        "role": "state_info",
                        "class_name": class_name,
                        "content": {
                            key: value
                            for key, value in vars(class_instance).items()
                            if not key.startswith("_")
                        },
                    }
                )
            if state_log:
                all_inference_log.append(state_log)

        if force_quit:
            break

    if base_handler_module.is_memory_prereq(test_entry_id):
        assert len(involved_instances) == 1, "Memory category should only involve one class."
        memory_instance = list(involved_instances.values())[0]
        memory_instance._flush_memory_to_local_file()

    metadata = {
        "input_token_count": total_input_token_count,
        "output_token_count": total_output_token_count,
        "latency": total_latency,
        "inference_log": all_inference_log,
    }
    if not all(
        all(content == "" for content in single_turn_reasoning_content)
        for single_turn_reasoning_content in all_reasoning_content
    ):
        metadata["reasoning_content"] = all_reasoning_content

    return all_model_response, metadata


def _make_entry_runtime(test_entry_id: str) -> SRIRuntime | None:
    """Per-entry runtime shared by SRI and/or CRI.

    When SRI is off but CRI is enabled the runtime is still needed to
    track ``state_by_domain`` + ``user_turn_texts`` for CRI's UR
    detector and state snapshot, so it is created in ``passive_log``
    mode: SRI observes but never intervenes (``.active`` is False), and
    ``observe_execution_results`` only early-returns when mode == 'off'.
    """
    sri_mode = _sri_mode()
    if sri_mode != "off":
        return SRIRuntime(test_entry_id, sri_mode, _sri_variant())
    if _cri_enabled():
        return SRIRuntime(test_entry_id, "passive_log", _sri_variant())
    return None


def _apply_sri_patches(base_handler_module: Any) -> None:
    """Install the multi-turn driver that hosts SRI and CRI.

    Replaces ``BaseHandler.inference_multi_turn_{FC,prompting}`` with
    wrappers that instantiate ``SRIRuntime`` for the duration of each
    multi-turn test entry and delegate to ``_inference_multi_turn``.
    """

    sri_mode = _sri_mode()
    if sri_mode == "off" and not _cri_enabled():
        return

    base_handler_cls = base_handler_module.BaseHandler
    if not getattr(base_handler_cls.inference_multi_turn_FC, "_bfclcalib_sri_patched", False):
        original_fc = base_handler_cls.inference_multi_turn_FC

        def _sri_fc(self, test_entry, include_input_log, exclude_state_log):
            test_entry_id = str(test_entry.get("id", ""))
            if "multi_turn" not in test_entry_id:
                return original_fc(self, test_entry, include_input_log, exclude_state_log)
            previous_sri_runtime = getattr(_SRI_CONTEXT, "runtime", None)
            _SRI_CONTEXT.runtime = _make_entry_runtime(test_entry_id)
            try:
                return _inference_multi_turn(
                    self,
                    test_entry,
                    include_input_log,
                    exclude_state_log,
                    base_handler_module=base_handler_module,
                    fc_mode=True,
                )
            finally:
                if previous_sri_runtime is None:
                    try:
                        delattr(_SRI_CONTEXT, "runtime")
                    except AttributeError:
                        pass
                else:
                    _SRI_CONTEXT.runtime = previous_sri_runtime

        _sri_fc._bfclcalib_sri_patched = True  # type: ignore[attr-defined]
        base_handler_cls.inference_multi_turn_FC = _sri_fc

    if not getattr(base_handler_cls.inference_multi_turn_prompting, "_bfclcalib_sri_patched", False):
        original_prompting = base_handler_cls.inference_multi_turn_prompting

        def _sri_prompting(self, test_entry, include_input_log, exclude_state_log):
            test_entry_id = str(test_entry.get("id", ""))
            if "multi_turn" not in test_entry_id:
                return original_prompting(self, test_entry, include_input_log, exclude_state_log)
            previous_sri_runtime = getattr(_SRI_CONTEXT, "runtime", None)
            _SRI_CONTEXT.runtime = _make_entry_runtime(test_entry_id)
            try:
                return _inference_multi_turn(
                    self,
                    test_entry,
                    include_input_log,
                    exclude_state_log,
                    base_handler_module=base_handler_module,
                    fc_mode=False,
                )
            finally:
                if previous_sri_runtime is None:
                    try:
                        delattr(_SRI_CONTEXT, "runtime")
                    except AttributeError:
                        pass
                else:
                    _SRI_CONTEXT.runtime = previous_sri_runtime

        _sri_prompting._bfclcalib_sri_patched = True  # type: ignore[attr-defined]
        base_handler_cls.inference_multi_turn_prompting = _sri_prompting
