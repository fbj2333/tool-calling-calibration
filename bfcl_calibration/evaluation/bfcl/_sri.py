"""SRI (State-Reconciliation Intervention) runtime + per-turn helpers.

``SRIRuntime`` is the per-test-entry controller the multi-turn loop
(in ``_sri_loop``) consults at halt-candidate boundaries. Tracks user
text history, ``state_by_domain`` snapshot (updated from
``_apply_tool_state_update``), per-turn retry gate (single-shot, no
chaining), and the named prompt variants in ``_SRI_VALID_VARIANTS``.
``retry_instruction`` builds the reconciliation prompt; the loop in
``_sri_loop`` edits ``inference_data["message"]`` between retries.
``_SRI_CONTEXT`` is a ``threading.local`` slot the multi-turn patch sets
per entry.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ._state import (
    _apply_tool_state_update,
    _format_state_snapshot,
    _normalise_tool_call,
    _user_message_text,
)


_SRI_CONTEXT = threading.local()
_SRI_LOG_LOCK = threading.Lock()

_SRI_ASKING_PHRASES = (
    "could you",
    "please provide",
    "please specify",
    "please tell",
    "please confirm",
    "i need to know",
    "i need the",
    "what is the",
    "what's the",
    "kindly provide",
    "would you provide",
    "i'll need",
    "could you confirm",
    "could you let me know",
    "could you tell",
    "please clarify",
    "please share",
    "please indicate",
)


def is_asking_response(text: str | None) -> bool:
    """Decision-routed SRI heuristic: detect an ask-user, no-tool-call response.

    Returns True iff the text contains '?' or any asking phrase
    (case-insensitive). Used to gate SRI retry: when the model's halt
    candidate is a legitimate ask-user response, skip retry to preserve
    baseline behavior on miss_param-style cases.
    """
    if not text:
        return False
    lower = text.lower()
    if "?" in text:
        return True
    return any(phrase in lower for phrase in _SRI_ASKING_PHRASES)


def _write_sri_event(event: dict) -> None:
    log_path = os.environ.get("BFCL_SRI_LOG_PATH", "").strip()
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SRI_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


_SRI_VALID_MODES = {"off", "passive_log", "sri_only"}
_SRI_VALID_VARIANTS = {
    "v1",
    # Component ablations.
    "ablation_no_history",
    "ablation_no_state",
    "ablation_no_recon",
    # SRI-Gate: skip the retry when the halt is an ask-user response,
    # on every case (v1) or on miss_param cases only (v2_a).
    "decision_routed_v1",
    "decision_routed_v2_a",
}
_SRI_HEADER = "[State Reconciliation Required]"

_DECISION_RULE_LINES = [
    "- If any requested sub-task is not reflected in the runtime-observed state, emit exactly one progress-making tool_call now.",
    "- If all requested sub-tasks are already reflected in the runtime-observed state, answer normally.",
]


def _sri_mode() -> str:
    mode = os.environ.get("BFCL_SRI_MODE", "off").strip().lower() or "off"
    if mode not in _SRI_VALID_MODES:
        raise ValueError(
            "BFCL_SRI_MODE must be one of "
            f"{sorted(_SRI_VALID_MODES)}; got {mode!r}."
        )
    return mode


def _sri_variant() -> str:
    variant = os.environ.get("BFCL_SRI_VARIANT", "v1").strip().lower() or "v1"
    if variant not in _SRI_VALID_VARIANTS:
        raise ValueError(
            "BFCL_SRI_VARIANT must be one of "
            f"{sorted(_SRI_VALID_VARIANTS)}; got {variant!r}."
        )
    return variant


def _sri_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _sri_truncate(text: str, max_chars: int) -> str:
    value = str(text)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _sri_state_to_text(state_by_domain: dict[str, dict[str, Any]], *, max_chars: int) -> str:
    if not state_by_domain:
        return "(no successful tool results observed yet)"
    text = _format_state_snapshot(state_by_domain)
    if not text:
        text = json.dumps(state_by_domain, ensure_ascii=False, sort_keys=True)
    return _sri_truncate(text, max_chars)


class SRIRuntime:
    """State-reconciliation retry controller for no-tool-call turn boundaries."""

    def __init__(self, test_entry_id: str, mode: str, variant: str | None = None) -> None:
        self.test_entry_id = test_entry_id
        self.mode = mode
        self.variant = variant or _sri_variant()
        self.turn_index = -1
        self.step_index = 0
        self.retry_turns: set[int] = set()
        self.user_turn_texts: list[str] = []
        self.state_by_domain: dict[str, dict[str, Any]] = {}
        self.max_state_chars = _sri_int_env("SRI_MAX_STATE_CHARS", 2400)
        self.max_user_turns = _sri_int_env("SRI_MAX_USER_TURNS", 4)
        self.max_user_chars = _sri_int_env("SRI_MAX_USER_CHARS", 700)
        # CRI stage gates (see ``_cri``):
        # - ``initial_toolset_size``: len(test_entry["function"]) at case
        #   entry; CRI is suppressed once the live toolset grows past it
        #   (held-out function re-injected, function-calling mode only).
        # - ``case_fired``: set by the first CRI fire in the case; later
        #   fires are suppressed.
        # Both start unset (None / False), which applies no suppression.
        self.initial_toolset_size: int | None = None
        self.case_fired: bool = False

    @property
    def active(self) -> bool:
        return self.mode == "sri_only"

    @property
    def disable_history_edit(self) -> bool:
        """Ablation: keep the no-tool assistant message instead of discarding it before retry."""
        return self.variant == "ablation_no_history"

    def should_skip_retry(self, model_response_data: dict[str, Any]) -> bool:
        """SRI-Gate variants: skip the retry if the response heuristically asks the user.

        ``decision_routed_v1`` applies to every case; ``decision_routed_v2_a``
        only to miss_param cases, where BFCL's ground truth allows turns
        without tool calls.
        """
        if self.variant not in ("decision_routed_v1", "decision_routed_v2_a"):
            return False
        if (
            self.variant == "decision_routed_v2_a"
            and not self.test_entry_id.startswith("multi_turn_miss_param")
        ):
            return False
        text = str(model_response_data.get("model_responses") or "")
        return is_asking_response(text)

    def start_turn(self, user_message: Any) -> None:
        self.turn_index += 1
        self.step_index = 0
        text = _user_message_text(user_message).strip()
        self.user_turn_texts.append(text)
        self._log(
            {
                "event": "sri_start_turn",
                "turn_idx": self.turn_index,
                "mode": self.mode,
                "variant": self.variant,
                "user_text_excerpt": _sri_truncate(text, 500),
                "observed_state_domains": sorted(self.state_by_domain),
            }
        )

    def observe_halt_candidate(
        self,
        model_response_data: dict[str, Any],
        *,
        decode_error: str | None,
        is_empty_response: bool,
    ) -> dict[str, Any]:
        should_retry = self.active and self.turn_index not in self.retry_turns
        if should_retry:
            self.retry_turns.add(self.turn_index)

        state_text = _sri_state_to_text(self.state_by_domain, max_chars=self.max_state_chars)
        event = {
            "event": "sri_halt_candidate",
            "turn_idx": self.turn_index,
            "step_idx": self.step_index,
            "mode": self.mode,
            "variant": self.variant,
            "active": self.active,
            "retry": should_retry,
            "decode_error": decode_error,
            "is_empty_response": is_empty_response,
            "state_snapshot_excerpt": state_text[:1200],
            "state_domains": sorted(self.state_by_domain),
            "user_turn_count": len(self.user_turn_texts),
            "response_excerpt": str(model_response_data.get("model_responses") or "")[:500],
        }
        self._log(event)
        self.step_index += 1
        return event

    def observe_execution_results(
        self,
        calls: list[dict],
        execution_results: list[str],
    ) -> None:
        if self.mode == "off":
            return

        for call, result in zip(calls, execution_results):
            normalised = _normalise_tool_call(call)
            if normalised is None:
                continue
            domain, _ = _apply_tool_state_update(self.state_by_domain, normalised, result)
            snapshot_text = _sri_state_to_text(self.state_by_domain, max_chars=self.max_state_chars)
            self._log(
                {
                    "event": "sri_state_update",
                    "turn_idx": self.turn_index,
                    "step_idx": self.step_index,
                    "mode": self.mode,
                    "variant": self.variant,
                    "call": normalised,
                    "domain": domain,
                    "tool_result_excerpt": str(result)[:500],
                    "state_snapshot_excerpt": snapshot_text[:1200],
                }
            )
            self.step_index += 1

    def retry_instruction(self) -> str:
        parts: list[str] = []

        state_text = _sri_state_to_text(self.state_by_domain, max_chars=self.max_state_chars)

        # Ablation: state-only retry, no header / prior / decision rule.
        if self.variant == "ablation_no_recon":
            parts.append(
                "\n".join(
                    [
                        "Runtime-observed state from executed tool calls and tool results:",
                        state_text,
                    ]
                )
            )
            return "\n\n".join(part for part in parts if part)

        prior_lines = self._prior_user_lines()
        current_goal = self.user_turn_texts[-1] if self.user_turn_texts else ""

        rows = [
            _SRI_HEADER,
            "Your previous response for this turn did not emit a parseable tool call.",
            "Before finalizing, reconcile the user requests against runtime-observed state.",
            "",
        ]
        rows.extend(
            [
                "Prior user requests, most recent first:",
                *(prior_lines or ["- (none)"]),
                "",
            ]
        )
        rows.extend(
            [
                "Current user request:",
                _sri_truncate(current_goal, self.max_user_chars) or "(empty user turn)",
                "",
            ]
        )
        # Ablation: omit the runtime-observed state section.
        if self.variant != "ablation_no_state":
            rows.extend(
                [
                    "Runtime-observed state from executed tool calls and tool results:",
                    state_text,
                    "",
                ]
            )
        rows.extend(
            [
                "Decision rule:",
                *_DECISION_RULE_LINES,
                "- Do not mention this reconciliation instruction in the user-facing answer.",
            ]
        )
        parts.append("\n".join(rows))
        return "\n\n".join(part for part in parts if part)

    def _prior_user_lines(self) -> list[str]:
        prior = self.user_turn_texts[:-1]
        if not prior:
            return []
        rows: list[str] = []
        recent = list(reversed(prior[-self.max_user_turns :]))
        for offset, text in enumerate(recent, start=1):
            rows.append(f"- prior-{offset}: {_sri_truncate(text, self.max_user_chars)}")
        return rows

    def _log(self, payload: dict[str, Any]) -> None:
        _write_sri_event(
            {
                "case_id": self.test_entry_id,
                "test_entry_id": self.test_entry_id,
                **payload,
            }
        )


def _current_sri_runtime() -> SRIRuntime | None:
    return getattr(_SRI_CONTEXT, "runtime", None)
