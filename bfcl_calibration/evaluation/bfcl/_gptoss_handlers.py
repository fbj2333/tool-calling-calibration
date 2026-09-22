"""gpt-oss function-calling handler for BFCL.

``_make_gptoss_fc_handler(oss_handler_cls)`` builds ``GPTOSSFCHandler``, a
subclass of upstream's ``OSSHandler`` that queries the vLLM server through the
OpenAI Responses API (gpt-oss's native function-calling protocol) and converts
its output to BFCL's tool-call format. The I/O helpers it uses are in
``_gptoss``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from ._gptoss import (
    _gptoss_canonical_tool_text,
    _gptoss_extract_tool_calls,
    _gptoss_response_tool_specs,
    _gptoss_responses_input,
    _gptoss_responses_output,
)


def _make_gptoss_fc_handler(oss_handler_cls: Any) -> Any:
    from bfcl_eval.model_handler.utils import convert_to_function_call

    def _fc_pre_query(self, test_entry: dict) -> dict:
        return {"message": [], "function": test_entry["function"]}

    def _responses_query(self, inference_data: dict):
        functions: list[dict] = inference_data["function"]
        messages: list[dict] = inference_data["message"]
        response_input = _gptoss_responses_input(messages)
        tools = _gptoss_response_tool_specs(functions) if functions else None
        inference_data["inference_input_log"] = {
            "input": response_input,
            "tools": tools or [],
        }

        kwargs = {
            "model": self.model_path_or_id,
            "input": response_input,
            "temperature": self.temperature,
            "max_output_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = tools
            tool_choice = os.environ.get("GPTOSS_TOOL_CHOICE")
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        reasoning_effort = os.environ.get("GPTOSS_REASONING_EFFORT")
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        start_time = time.time()
        api_response = self.client.responses.create(**kwargs)
        end_time = time.time()
        return api_response, end_time - start_time

    def _fc_query(self, inference_data: dict):
        return _responses_query(self, inference_data)

    def _fc_parse_response(self, api_response: Any) -> dict:
        reasoning_content, cleaned_response, extracted_tool_calls = _gptoss_responses_output(api_response)
        if extracted_tool_calls:
            cleaned_response = _gptoss_canonical_tool_text(extracted_tool_calls)
            tool_call_ids = []
            history_tool_calls = []
            for index, call in enumerate(extracted_tool_calls, start=1):
                call_id = call.get("id") or f"call_{index}"
                tool_call_ids.append(call_id)
                history_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(
                                call.get("arguments", {}),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            history_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": history_tool_calls,
            }
        else:
            tool_call_ids = []
            history_message = {
                "role": "assistant",
                "content": cleaned_response,
            }
        history_message["reasoning_content"] = reasoning_content
        usage = getattr(api_response, "usage", None)
        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": history_message,
            "tool_call_ids": tool_call_ids,
            "input_token": getattr(usage, "input_tokens", 0) if usage else 0,
            "output_token": getattr(usage, "output_tokens", 0) if usage else 0,
        }

    def _fc_add_assistant(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    def _fc_add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, tool_call_id in zip(
            execution_results,
            model_response_data.get("tool_call_ids", []),
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": execution_result,
                }
            )
        return inference_data

    def _fc_decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = _gptoss_extract_tool_calls(result)
        if not tool_calls:
            tool_calls = _gptoss_extract_tool_calls(_gptoss_canonical_tool_text([]) + result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call.get("arguments", {}).items()}}
            for call in tool_calls
        ]

    def _fc_decode_execute(self, result, has_tool_call_tag):
        tool_calls = _gptoss_extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = [{item["name"]: item.get("arguments", {})} for item in tool_calls]
        return convert_to_function_call(decoded_result)

    for method in (
        _fc_pre_query,
        _fc_query,
        _fc_parse_response,
        _fc_add_assistant,
        _fc_add_execution_results,
        _fc_decode_ast,
        _fc_decode_execute,
    ):
        method.__override__ = True  # type: ignore[attr-defined]

    return type(
        "GPTOSSFCHandler",
        (oss_handler_cls,),
        {
            "_pre_query_processing_prompting": _fc_pre_query,
            "_query_prompting": _fc_query,
            "_parse_query_response_prompting": _fc_parse_response,
            "_add_assistant_message_prompting": _fc_add_assistant,
            "_add_execution_results_prompting": _fc_add_execution_results,
            "decode_ast": _fc_decode_ast,
            "decode_execute": _fc_decode_execute,
        },
    )
