"""Tool-call extraction + per-tool state-extraction rules.

Two related utility groups live here:

1. **Tool-call utilities** — ``_normalise_tool_call``,
   ``_extract_xml_tool_calls`` / ``_extract_json_tool_calls``,
   ``_response_data_tool_calls``, ``_user_message_text`` and the
   small ``_is_error_tool_result`` helper — read
   tool calls out of model responses (XML ``<tool_call>`` or JSON
   tool array shapes).
2. **State-snapshot helpers** — ``_format_state_value`` /
   ``_flatten_state`` / ``_format_state_snapshot`` render the
   ``state_by_domain`` snapshot into the text SRI's retry prompt and
   CRI's detectors consume; ``_tool_result_is_error``,
   ``_TOOL_STATE_RULES``, ``_apply_tool_state_update`` and the per-tool
   updaters (``_vehicle_engine_update``, ``_vehicle_lock_update``,
   ``_filesystem_mv_update``, ``_message_send_update``) maintain the
   snapshot from observed (call, result) pairs.

The extraction table covers the BFCL multi-turn API surface that SRI's
state section and CRI's detectors consume; new domains/tools can be added
to ``_TOOL_STATE_RULES``.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any


_TOOL_RESULT_ERROR_RE = re.compile(
    r"error during execution|[\"']?error[\"']?\s*:|\b("
    r"failed|failure|invalid|not found|does not exist|exception|"
    r"permission denied|not allowed|cannot|can't)\b",
    re.IGNORECASE,
)


def _normalise_tool_call(call: dict) -> dict | None:
    function = call.get("function", call)
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments", {}) if isinstance(function, dict) else {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    result = {"name": name, "arguments": arguments}
    if isinstance(call.get("id"), str):
        result["id"] = call["id"]
    return result


def _tool_call_signature(call: dict) -> str:
    return json.dumps(
        {
            "name": call.get("name", ""),
            "arguments": call.get("arguments", {}),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _extract_xml_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    for raw_call in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
        try:
            call = json.loads(raw_call.strip())
        except Exception:
            continue
        normalised = _normalise_tool_call(call) if isinstance(call, dict) else None
        if normalised is not None:
            calls.append(normalised)
    return calls


def _extract_json_tool_calls(text: str) -> list[dict]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates = [stripped]
    if ";" in stripped:
        candidates.extend(part.strip() for part in stripped.split(";") if part.strip())

    calls: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        raw_calls = parsed if isinstance(parsed, list) else [parsed]
        for raw_call in raw_calls:
            normalised = _normalise_tool_call(raw_call) if isinstance(raw_call, dict) else None
            if normalised is None:
                continue
            signature = _tool_call_signature(normalised)
            if signature in seen:
                continue
            seen.add(signature)
            calls.append(normalised)
    return calls


def _parse_execution_strings(decoded: list[str]) -> list[dict]:
    """Parse BFCL handler ``decode_execute`` outputs into normalised calls.

    BFCL handlers expose a post-handler canonical execution form such as
    ``"mkdir(dir_name='temp')"`` or ``"mv(source='a', destination='b')"``.
    That form is generated after each model-specific decoder has handled raw
    output quirks, so it is the most scorer-aligned surface CRI can read across
    model families. Entries that do not parse as a keyword-only call are
    skipped.
    """
    calls: list[dict] = []
    for item in decoded:
        if not isinstance(item, str) or not item.strip():
            continue
        try:
            expression = ast.parse(item.strip(), mode="eval")
        except SyntaxError:
            continue

        node = expression.body
        if not isinstance(node, ast.Call) or node.args:
            continue

        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue

        arguments: dict[str, Any] = {}
        valid = True
        for keyword in node.keywords:
            if keyword.arg is None:
                valid = False
                break
            try:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError, SyntaxError):
                valid = False
                break
        if not valid:
            continue

        calls.append({"name": name, "arguments": arguments})
    return calls


def _is_error_tool_result(result: str) -> bool:
    return bool(_TOOL_RESULT_ERROR_RE.search(str(result)))


def _response_data_tool_calls(model_response_data: dict) -> list[dict]:
    # Prefer BFCL's post-handler canonical execution strings when present.
    # These are produced by ``decode_execute`` before CRI is consulted in the
    # multi-turn driver, so they already reflect each model family's raw output
    # grammar and match the scorer-visible execution surface.
    decoded_responses = model_response_data.get("model_responses_decoded")
    if isinstance(decoded_responses, list) and decoded_responses:
        parsed = _parse_execution_strings(decoded_responses)
        if parsed:
            return parsed

    # Closed-API OpenAI-compatible handler shape: ``model_responses`` is a
    # ``list[dict]`` where each dict has a single ``{name: args_json_string}``
    # entry (see ``bfcl_eval/model_handler/api_inference/openai_completion.py``
    # ``_parse_query_response_FC``), the shape API models run through
    # ``scripts/run_bfcl_api.py`` produce. The XML / JSON / chat-history
    # extractors below do not read it.
    raw_responses = model_response_data.get("model_responses")
    if isinstance(raw_responses, list):
        normalized: list[dict] = []
        for entry in raw_responses:
            if not isinstance(entry, dict) or len(entry) != 1:
                continue
            name, args = next(iter(entry.items()))
            if not isinstance(name, str):
                continue
            if isinstance(args, str):
                try:
                    args_obj = json.loads(args)
                except (ValueError, TypeError):
                    args_obj = {}
            elif isinstance(args, dict):
                args_obj = args
            else:
                args_obj = {}
            normalized.append({"name": name, "arguments": args_obj})
        if normalized:
            return normalized

    calls = _extract_xml_tool_calls(str(model_response_data.get("model_responses", "")))
    if calls:
        return calls
    calls = _extract_json_tool_calls(str(model_response_data.get("model_responses", "")))
    if calls:
        return calls

    history_message = model_response_data.get("model_responses_message_for_chat_history")
    if not isinstance(history_message, dict):
        return []
    raw_calls = history_message.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []
    return [
        normalised
        for call in raw_calls
        if isinstance(call, dict)
        if (normalised := _normalise_tool_call(call)) is not None
    ]


def _user_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("content", ""))
    if isinstance(message, list):
        parts = []
        for item in message:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(message)


# --------------------------------------------------------------------------- #
# State-snapshot helpers — shared by SRI's retry prompt and CRI's detectors.  #
# --------------------------------------------------------------------------- #


def _format_state_value(value: Any, *, max_len: int = 500) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _flatten_state(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_state(child_prefix, value[key]))
        return rows
    return [(prefix, value)]


def _format_state_snapshot(state_by_domain: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for domain in sorted(state_by_domain):
        for key, value in _flatten_state(domain, state_by_domain[domain]):
            if key:
                lines.append(f"{key}: {_format_state_value(value)}")
    return "\n".join(lines)


def _tool_result_is_error(result: Any) -> bool:
    """Heuristic: treat dict-shaped error markers and our generic regex as errors."""
    text = str(result)
    if _is_error_tool_result(text):
        return True
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    if isinstance(parsed, dict) and "error" in parsed:
        return True
    return False


# Tool-name → (domain, update-callable). The callable receives the tool call's
# normalised arguments dict and the raw result string and returns a partial
# state update merged into ``state_by_domain[domain]``.
def _vehicle_engine_update(args: dict[str, Any], _result: Any) -> dict[str, Any]:
    mode = str(args.get("ignitionMode", "")).upper()
    return {"engine_state": "running" if mode == "START" else "stopped"}


def _vehicle_lock_update(args: dict[str, Any], _result: Any) -> dict[str, Any]:
    return {"doors_locked": not bool(args.get("unlock", False))}


def _filesystem_mv_update(args: dict[str, Any], _result: Any) -> dict[str, Any]:
    return {
        "last_path_transition": {
            "op": "mv",
            "source": str(args.get("source", "")),
            "destination": str(args.get("destination", "")),
        }
    }


def _message_send_update(args: dict[str, Any], _result: Any) -> dict[str, Any]:
    receiver = str(args.get("receiver_id") or args.get("recipient") or "")
    payload = {
        "messages_sent": [
            {
                "receiver_id": receiver,
                "message": str(args.get("message", "")),
            }
        ]
    }
    return payload


_TOOL_STATE_RULES: dict[str, tuple[str, Any]] = {
    "startengine": ("vehicle", _vehicle_engine_update),
    "lockdoors": ("vehicle", _vehicle_lock_update),
    "mv": ("filesystem", _filesystem_mv_update),
    "send_message": ("message", _message_send_update),
}


def _domain_for_tool(tool_name: str) -> str:
    rule = _TOOL_STATE_RULES.get(tool_name.lower())
    return rule[0] if rule else "generic"


def _apply_tool_state_update(
    state_by_domain: dict[str, dict[str, Any]],
    call: dict[str, Any],
    result: Any,
) -> tuple[str, dict[str, Any]]:
    """Update ``state_by_domain`` in place from one (call, result) pair.

    Returns ``(domain, updated_domain_state)``. Errors leave state untouched.
    """
    name = str(call.get("name", ""))
    domain = _domain_for_tool(name)
    if _tool_result_is_error(result):
        return domain, state_by_domain.get(domain, {})

    rule = _TOOL_STATE_RULES.get(name.lower())
    domain_state = dict(state_by_domain.get(domain, {}))
    if rule is None:
        # Unknown tool: surface domain presence only, no per-key evidence.
        if domain not in state_by_domain:
            state_by_domain[domain] = domain_state
        return domain, domain_state

    args = call.get("arguments", {}) if isinstance(call.get("arguments"), dict) else {}
    update = rule[1](args, result)
    for key, value in update.items():
        if isinstance(value, list) and isinstance(domain_state.get(key), list):
            domain_state[key] = list(domain_state[key]) + list(value)
        else:
            domain_state[key] = value
    state_by_domain[domain] = domain_state
    return domain, domain_state
