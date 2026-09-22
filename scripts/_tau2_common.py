"""Shared helpers for the tau2-bench scripts."""
from __future__ import annotations

from typing import Any


def failed_action_count(reward_info: dict[str, Any]) -> int:
    checks = reward_info.get("action_checks") or []
    if not isinstance(checks, list):
        return 0
    return sum(
        1
        for check in checks
        if isinstance(check, dict)
        and (check.get("action_match") is False or check.get("action_reward") == 0)
    )


def _truthy_error(value: Any) -> bool:
    if value in (None, False, "", [], {}):
        return False
    return True


def is_infrastructure_error(sim: dict[str, Any]) -> bool:
    """Best-effort scored-only guard for tau2 runs.

    The project tables use scored simulations. We exclude explicit run/runtime
    errors while keeping normal task failures, including max-turn failures, in
    the denominator.
    """

    for key in (
        "error",
        "exception",
        "traceback",
        "infrastructure_error",
        "runtime_error",
    ):
        if _truthy_error(sim.get(key)):
            return True

    reward_info = sim.get("reward_info")
    if isinstance(reward_info, dict):
        for key in ("error", "exception", "traceback", "infrastructure_error"):
            if _truthy_error(reward_info.get(key)):
                return True
        basis = str(reward_info.get("reward_basis") or "").lower()
        if "infrastructure" in basis or "runtime_error" in basis:
            return True

    status_text = " ".join(
        str(sim.get(key) or "")
        for key in ("status", "termination_reason", "failure_reason")
    ).lower()
    return "infrastructure" in status_text or "runtime_error" in status_text


def is_scored_simulation(sim: dict[str, Any]) -> bool:
    reward_info = sim.get("reward_info")
    return isinstance(reward_info, dict) and not is_infrastructure_error(sim)


def pass_at_one_hit(reward_info: dict[str, Any]) -> bool:
    return reward_info.get("reward") == 1
