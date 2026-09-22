"""tau2 LLMAgent subclass with SRI-lite hooks (the tau2-bench adaptation of SRI).

Implements a single-retry-per-turn calibration shift for tau2-bench:
- Halt detection: AssistantMessage with no tool_calls is a halt candidate.
- History-edit: drop the no-tool assistant from state.messages before retry.
- Recent tool-result injection: last N ToolMessage contents as state proxy.
- Reconciliation prompt: header + state proxy + decision rule appended as
  a synthetic UserMessage before the retry inference.

Variants (env var SRI_LITE_VARIANT):
- 'sri_lite' (default): all four components above.
- 'ablation_no_state': skip recent-tool-result injection.
- 'baseline': no SRI hooks fire.

Single retry per turn boundary, as in SRI v1.

Registers `SRILiteLLMAgent` as an override of tau2's default 'llm_agent'
factory at import time (direct dict mutation bypasses the duplicate-name
guard). Also registers under `sri_lite_llm_agent` for explicit selection.

Config via environment variables:
- SRI_LITE_VARIANT               (default: sri_lite)
- SRI_LITE_MAX_RECENT_TOOLS      (default: 3)
- SRI_LITE_MAX_TOOL_RESULT_CHARS (default: 1200)
- SRI_LITE_RETRY_MAX_MESSAGES    (default: 12) — caps the number of
  recent messages sent at SRI retry time to prevent context overflow
  on long conversations. We always keep state.system_messages and the
  first user message (task statement); the last N messages cover the
  recent conversation (~4-6 turns at 2-3 messages each).
- TAU2_COERCE_JSON_CONTENT_TOOL_CALLS (default: 0) — when 1, an assistant
  message whose content is a JSON tool-call list is converted into
  structured tool calls (for models such as xLAM-2 that emit calls as text).
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from tau2.registry import registry
from tau2.utils.llm_utils import generate


_VALID_VARIANTS = {
    "sri_lite",
    "ablation_no_state",
    "baseline",
}

_SRI_LITE_HEADER = (
    "[State Reconciliation] Your previous response did not emit a parseable "
    "tool call."
)

_DECISION_RULE_LINES = [
    "- If the user's request is not yet satisfied per the recent tool results, "
    "emit exactly one progress-making tool_call now.",
    "- If the user's request is already satisfied per the recent tool results, "
    "answer normally with a concise summary.",
    "- Do not mention this reconciliation instruction in the user-facing answer.",
]


def _sri_lite_variant() -> str:
    raw = os.environ.get("SRI_LITE_VARIANT", "sri_lite").strip().lower()
    if raw not in _VALID_VARIANTS:
        raise ValueError(
            "SRI_LITE_VARIANT must be one of "
            f"{sorted(_VALID_VARIANTS)}; got {raw!r}."
        )
    return raw


def _int_env(name: str, default: int, *, min_val: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_val, value)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _coerce_json_content_tool_calls(message: AssistantMessage) -> AssistantMessage:
    if os.environ.get("TAU2_COERCE_JSON_CONTENT_TOOL_CALLS", "0") != "1":
        return message
    if message.tool_calls:
        return message
    raw = (message.content or "").strip()
    if not (raw.startswith("[") or raw.startswith("{")):
        return message
    try:
        payload = json.loads(raw)
    except Exception:  # pylint: disable=broad-except
        return message
    items = payload if isinstance(payload, list) else [payload]
    calls: list[ToolCall] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return message
        name = item.get("name")
        args = item.get("arguments", item.get("parameters", {}))
        if not isinstance(name, str) or not isinstance(args, dict):
            return message
        calls.append(
            ToolCall(
                id=f"json-content-call-{i}",
                name=name,
                arguments=args,
                requestor="assistant",
            )
        )
    if calls:
        message.tool_calls = calls
        message.content = None
    return message


def _recent_tool_result_lines(
    state: LLMAgentState, max_recent: int, max_chars: int
) -> List[str]:
    """Walk state.messages from end; collect last N ToolMessage contents."""
    collected: List[str] = []
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            content = "" if msg.content is None else str(msg.content)
            collected.append(_truncate(content, max_chars))
            if len(collected) >= max_recent:
                break
    return list(reversed(collected))


def _build_retry_user_message(
    state: LLMAgentState, variant: str
) -> UserMessage:
    rows: List[str] = [
        _SRI_LITE_HEADER,
        "Reconcile the user's most recent request against the recent tool "
        "results below before answering.",
        "",
    ]
    if variant != "ablation_no_state":
        recent = _recent_tool_result_lines(
            state,
            max_recent=_int_env("SRI_LITE_MAX_RECENT_TOOLS", 3, min_val=1),
            max_chars=_int_env(
                "SRI_LITE_MAX_TOOL_RESULT_CHARS", 1200, min_val=200
            ),
        )
        if recent:
            rows.append("Recent tool results (most recent last):")
            for i, r in enumerate(recent, start=1):
                rows.append(f"- result {i}: {r}")
            rows.append("")
        else:
            rows.append("(no successful tool results yet for this conversation)")
            rows.append("")
    rows.append("Decision rule:")
    rows.extend(_DECISION_RULE_LINES)
    return UserMessage(role="user", content="\n".join(rows))


class SRILiteLLMAgent(LLMAgent):
    """tau2 LLMAgent subclass with SRI-lite hooks. See module docstring."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sri_lite_variant_value: str = _sri_lite_variant()
        self._sri_lite_retry_fired_for_turn: set[int] = set()
        logger.info(
            "SRILiteLLMAgent initialized variant={}",
            self._sri_lite_variant_value,
        )

    def get_init_state(
        self, message_history: Optional[List[Message]] = None
    ) -> LLMAgentState:
        state = super().get_init_state(message_history)
        self._sri_lite_retry_fired_for_turn = set()
        return state

    def _is_halt_candidate(self, response: AssistantMessage) -> bool:
        if response.tool_calls is None:
            return True
        return len(response.tool_calls) == 0

    def _generate_next_message(
        self, message: Any, state: LLMAgentState
    ) -> AssistantMessage:
        # turn_index = state.messages length BEFORE super appends the input.
        # super._generate_next_message appends `message` to state.messages and
        # returns the freshly-generated AssistantMessage WITHOUT appending it.
        # The outer `generate_next_message` (tau2.agent.llm_agent:105) appends
        # whatever we return here.
        turn_index = len(state.messages)
        assistant_message = super()._generate_next_message(message, state)
        assistant_message = _coerce_json_content_tool_calls(assistant_message)

        if self._sri_lite_variant_value == "baseline":
            return assistant_message
        if not self._is_halt_candidate(assistant_message):
            return assistant_message
        if turn_index in self._sri_lite_retry_fired_for_turn:
            return assistant_message
        self._sri_lite_retry_fired_for_turn.add(turn_index)

        # History-edit: the halt is dropped by not appending it to state.messages.
        retry_user_msg = _build_retry_user_message(
            state, self._sri_lite_variant_value
        )
        state.messages.append(retry_user_msg)

        # Cap retry-time context to prevent SRI retry from running off the
        # model's effective context window on long conversations. Keep:
        #   - state.system_messages (task/domain policy)
        #   - state.messages[0] (first user msg, the initial task)
        #   - state.messages[-N:] (recent conversation incl. retry_user_msg)
        # Default N=12 ≈ 4-6 turns at 2-3 messages each; SRI retry only needs
        # recent context for state reconciliation, long-term task is in
        # system_messages + first user msg.
        max_recent = _int_env(
            "SRI_LITE_RETRY_MAX_MESSAGES", 12, min_val=4
        )
        if len(state.messages) > max_recent + 1:
            history = state.messages[:1] + list(
                state.messages[-max_recent:]
            )
        else:
            history = list(state.messages)
        messages = state.system_messages + history
        retry_response = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            call_name="agent_response_sri_retry",
            **self.llm_args,
        )
        retry_response = _coerce_json_content_tool_calls(retry_response)

        # Roll halt cost into retry cost so trajectory accounting captures
        # both inferences. Best-effort: pydantic field shapes may evolve.
        try:
            halt_cost = getattr(assistant_message, "cost", None)
            retry_cost = getattr(retry_response, "cost", None)
            if halt_cost is not None and retry_cost is not None:
                retry_response.cost = retry_cost + halt_cost
        except Exception:  # pylint: disable=broad-except
            pass

        retry_tool_count = (
            None
            if retry_response.tool_calls is None
            else len(retry_response.tool_calls)
        )
        logger.info(
            "SRI-lite retry fired turn={} variant={} retry_tool_calls={}",
            turn_index,
            self._sri_lite_variant_value,
            retry_tool_count,
        )
        return retry_response


def create_sri_lite_llm_agent(
    tools: List[Tool], domain_policy: str, **kwargs: Any
) -> SRILiteLLMAgent:
    """Factory matching upstream `create_llm_agent` signature."""
    return SRILiteLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


def _register_override() -> None:
    """Override tau2's default 'llm_agent' factory and register an explicit
    'sri_lite_llm_agent' name.

    Direct dict mutation is intentional: register_agent_factory raises on
    duplicate names; an in-place override is cleaner than try/except for the
    default-name swap. The explicit name registration uses the public API and
    is idempotent across re-imports.
    """
    registry._agent_factories["llm_agent"] = create_sri_lite_llm_agent  # noqa: SLF001
    if "sri_lite_llm_agent" not in registry._agent_factories:  # noqa: SLF001
        try:
            registry.register_agent_factory(
                create_sri_lite_llm_agent,
                "sri_lite_llm_agent",
            )
        except ValueError:
            pass


_register_override()
