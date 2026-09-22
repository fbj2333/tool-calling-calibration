"""I/O helpers for the gpt-oss function-calling handler (``_gptoss_handlers``).

Tool specs for the Responses API, conversion of BFCL's chat history to
Responses-API input items, parsing of Responses-API output, and the
``<tool_call>`` text form in which tool calls are stored in BFCL results.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _normalise_json_schema_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("object" if key == "type" and item == "dict" else _normalise_json_schema_types(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalise_json_schema_types(item) for item in value]
    return value


def _gptoss_response_tool_specs(functions: list[dict]) -> list[dict]:
    tools = []
    for function in functions:
        clean_function = _normalise_json_schema_types(json.loads(json.dumps(function)))
        clean_function["type"] = "function"
        tools.append(clean_function)
    return tools


def _gptoss_extract_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []

    # Native gpt-oss / harmony tool call format:
    # <|start|>assistant to=functions.foo<|channel|>commentary json<|message|>{...}<|call|>
    native_pattern = re.compile(
        r"<\|start\|>assistant\s+to=functions\.([^<\s]+)"
        r"<\|channel\|>commentary(?:\s+\w+)?<\|message\|>"
        r"(.*?)(?:<\|call\|>|<\|end\|>|<\|return\|>|$)",
        re.DOTALL,
    )
    for name, raw_arguments in native_pattern.findall(text):
        try:
            payload = json.loads(raw_arguments.strip())
        except Exception:
            continue
        if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
            call = payload
        else:
            call = {"name": name, "arguments": payload}
        if isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            calls.append(call)

    # Also accept Qwen-style XML tags in case the model follows BFCL text
    # instructions rather than its native channel format.
    xml_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for raw_call in xml_pattern.findall(text):
        try:
            call = json.loads(raw_call.strip())
        except Exception:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments"), dict):
            calls.append(call)

    return calls


def _gptoss_canonical_tool_text(calls: list[dict]) -> str:
    return "\n".join(
        "<tool_call>\n"
        + json.dumps(
            {"name": call["name"], "arguments": call.get("arguments", {})},
            ensure_ascii=False,
        )
        + "\n</tool_call>"
        for call in calls
    )


def _gptoss_responses_input(messages: list[dict]) -> list[dict]:
    response_input: list[dict] = []
    fallback_tool_ids: list[str] = []
    fallback_counter = 0

    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role in {"system", "user"}:
            response_input.append({"role": role, "content": content})
        elif role == "assistant":
            tool_calls = message.get("tool_calls", [])
            for call in tool_calls:
                fallback_counter += 1
                call_id = call.get("id") or f"call_{fallback_counter}"
                fallback_tool_ids.append(call_id)
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                response_input.append(
                    {
                        "type": "function_call",
                        "id": f"fc_{call_id}",
                        "call_id": call_id,
                        "name": function["name"],
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
            if content and not tool_calls:
                response_input.append(
                    {
                        "type": "message",
                        "id": f"msg_{len(response_input) + 1}",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": content,
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                    }
                )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id and fallback_tool_ids:
                call_id = fallback_tool_ids.pop(0)
            if call_id:
                response_input.append(
                    {
                        "type": "function_call_output",
                        "id": f"fco_{call_id}",
                        "call_id": call_id,
                        "output": str(content),
                        "status": "completed",
                    }
                )
            else:
                response_input.append(
                    {
                        "role": "user",
                        "content": f"<tool_response>\n{content}\n</tool_response>",
                    }
                )
    return response_input


def _gptoss_responses_output(api_response: Any) -> tuple[str, str, list[dict]]:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict] = []

    for item in getattr(api_response, "output", []) or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            for part in getattr(item, "content", []) or []:
                text = getattr(part, "text", None)
                if text:
                    reasoning_parts.append(text)
            for part in getattr(item, "summary", []) or []:
                text = getattr(part, "text", None)
                if text:
                    reasoning_parts.append(text)
        elif item_type == "function_call":
            name = getattr(item, "name", None)
            raw_arguments = getattr(item, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except Exception:
                arguments = {}
            if isinstance(name, str) and isinstance(arguments, dict):
                tool_calls.append(
                    {
                        "id": getattr(item, "call_id", None),
                        "name": name,
                        "arguments": arguments,
                    }
                )
        elif item_type == "message":
            for part in getattr(item, "content", []) or []:
                text = getattr(part, "text", None)
                if text:
                    content_parts.append(text)

    output_text = getattr(api_response, "output_text", None)
    if output_text and not content_parts:
        content_parts.append(output_text)

    return "\n".join(reasoning_parts), "\n".join(content_parts).strip(), tool_calls
