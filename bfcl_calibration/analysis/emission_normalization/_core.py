"""Normalize non-canonical BFCL tool emissions before action classification."""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from typing import Any

from bfcl_calibration.analysis.action_classifier import is_tool_call_emission


_NO_MATCH = object()
_Normalizer = Callable[[Any], object]


def normalize_emission(emission: Any) -> Any:
    """Return a classifier-compatible emission without changing semantics.

    Canonical strings and natural-language text are returned unchanged. Only
    whole-emission tool-call structures are serialized to the canonical
    ``[{"name": ..., "arguments": ...}]`` JSON form.
    """
    for normalizer in _NORMALIZERS:
        normalized = normalizer(emission)
        if normalized is not _NO_MATCH:
            return normalized
    return emission


def _canonical_string_noop(emission: Any) -> object:
    if isinstance(emission, str) and is_tool_call_emission(emission):
        return emission
    return _NO_MATCH


def _structured_tool_call(emission: Any) -> object:
    calls = _calls_from_value(emission)
    if calls is None:
        return _NO_MATCH
    return _to_json(calls)


def _literal_tool_call_string(emission: Any) -> object:
    if not isinstance(emission, str) or is_tool_call_emission(emission):
        return _NO_MATCH

    body = _whole_fence_body(emission)
    if body is None:
        body = emission.strip()
    if not _looks_like_literal_tool_payload(body):
        return _NO_MATCH

    value = _parse_literal(body)
    if value is _NO_MATCH:
        return _NO_MATCH

    calls = _calls_from_value(value)
    if calls is None:
        return _NO_MATCH
    return _to_json(calls)


def _calls_from_value(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        if not value:
            return None
        calls: list[dict[str, Any]] = []
        for item in value:
            call = _call_from_dict(item)
            if call is None:
                return None
            calls.append(call)
        return calls

    if isinstance(value, dict):
        direct_call = _call_from_dict(value)
        if direct_call is not None:
            return [direct_call]
        return _calls_from_tools_dict(value)

    return None


def _call_from_dict(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    if set(value) == {"name", "arguments"}:
        return _named_call(value["name"], value["arguments"])
    if set(value) == {"name", "parameters"}:
        return _named_call(value["name"], value["parameters"])

    if set(value) == {"tool_call", "arguments"}:
        return _named_call(value["tool_call"], value["arguments"])
    if set(value) == {"action", "arguments"}:
        return _named_call(value["action"], value["arguments"])

    if len(value) == 1:
        name, arguments = next(iter(value.items()))
        return _named_call(name, arguments)

    return None


def _calls_from_tools_dict(value: dict[Any, Any]) -> list[dict[str, Any]] | None:
    if set(value) != {"tools", "arguments"}:
        return None
    tools = value["tools"]
    arguments_by_tool = value["arguments"]
    if not isinstance(tools, list) or not isinstance(arguments_by_tool, dict):
        return None

    calls: list[dict[str, Any]] = []
    for name in tools:
        if not isinstance(name, str) or name not in arguments_by_tool:
            return None
        arguments = arguments_by_tool[name]
        call = _named_call(name, arguments)
        if call is None:
            return None
        calls.append(call)
    return calls or None


def _named_call(name: Any, arguments: Any) -> dict[str, Any] | None:
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def _whole_fence_body(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None

    lines = stripped.splitlines()
    if not lines or not lines[0].startswith("```"):
        return None

    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    if not body:
        return None
    return "\n".join(body).strip()


def _looks_like_literal_tool_payload(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("[{") or stripped.startswith("{")


def _parse_literal(text: str) -> Any:
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except (SyntaxError, ValueError):
            continue
    return _NO_MATCH


def _to_json(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=True, separators=(",", ":"))


_NORMALIZERS: tuple[_Normalizer, ...] = (
    _canonical_string_noop,
    _structured_tool_call,
    _literal_tool_call_string,
)
