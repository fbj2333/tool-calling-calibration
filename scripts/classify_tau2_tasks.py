#!/usr/bin/env python3
"""Classify tau2-bench tasks for the paper's diagnostic framework.

Task-level gold covers TOOL_CALL / ASK / REFUSE. CONFIRM has no task-level
gold: Tab. 3 scores it per database write against the domain policy
(``judge_tau2_confirm.py``), so it is not a task label here.

Detection rules:

* ``REFUSE``    -- ``evaluation_criteria.actions == []``.
* ``TOOL_CALL`` -- ``evaluation_criteria.actions`` non-empty (READ or WRITE);
                  the framework's TOOL_CALL gold is "agent emits a tool
                  call", with no restriction on state-change.
* ``ASK``       -- case-insensitive substring scan of the ``user_scenario``
                  JSON for patterns instructing the simulated user to
                  withhold task-relevant information until queried.

Labels are multi-valued. REFUSE and TOOL_CALL are mutually exclusive by
construction; ASK is orthogonal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DOMAINS = ("airline", "retail")

# Case-insensitive substring scan over ``user_scenario`` JSON dump.
ASK_KEYWORDS: tuple[str, ...] = (
    "if the agent asks",
    "if the service agent asks",
    "when the agent asks",
    "when the service agent asks",
    "do not volunteer",
    "do not mention unless",
    "don't mention",
    "withhold",
    "only reveal",
    "wait until asked",
    "if asked",
    "will not say anything that is not asked",
    "won't say anything that is not asked",
    "do not reveal it",
)


def _is_refuse_required(task: dict[str, Any]) -> bool:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    return len(actions) == 0


def _is_tool_call_required(task: dict[str, Any]) -> bool:
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    return len(actions) > 0


def _is_ask_required(task: dict[str, Any]) -> bool:
    scen_blob = json.dumps(
        task.get("user_scenario") or {}, ensure_ascii=False
    ).lower()
    return any(kw in scen_blob for kw in ASK_KEYWORDS)


def classify_tasks(root: Path, domain: str) -> dict[str, list[str]]:
    tasks_path = root / "data" / "tau2" / "domains" / domain / "tasks.json"
    tasks = json.loads(tasks_path.read_text())
    out: dict[str, list[str]] = {}
    for task in tasks:
        labels: list[str] = []
        if _is_refuse_required(task):
            labels.append("REFUSE")
        if _is_tool_call_required(task):
            labels.append("TOOL_CALL")
        if _is_ask_required(task):
            labels.append("ASK")
        out[str(task["id"])] = labels
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--tau2-root",
        required=True,
        type=Path,
        help="Path to a sierra-research/tau2-bench checkout (commit a03b7910).",
    )
    ap.add_argument(
        "--domain",
        choices=("airline", "retail", "all"),
        default="all",
    )
    ap.add_argument("--output-json", required=True, type=Path)
    args = ap.parse_args()

    domains = DOMAINS if args.domain == "all" else (args.domain,)
    result = {dom: classify_tasks(args.tau2_root.resolve(), dom) for dom in domains}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    for dom in domains:
        labels_per_task = result[dom]
        n_total = len(labels_per_task)
        n_refuse = sum(1 for v in labels_per_task.values() if "REFUSE" in v)
        n_ask = sum(1 for v in labels_per_task.values() if "ASK" in v)
        n_tool = sum(1 for v in labels_per_task.values() if "TOOL_CALL" in v)
        n_multi = sum(1 for v in labels_per_task.values() if len(v) > 1)
        print(
            f"{dom}: n={n_total}  REFUSE={n_refuse}  ASK={n_ask}  "
            f"TOOL_CALL={n_tool}  multi-label={n_multi}"
        )
    print(f"Wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
