#!/usr/bin/env python3
"""ASK-class GAR / Acc on tau2-bench under the audited ASK task subsets.

tau2-bench has no ASK annotation; ``classify_tau2_tasks.py`` selects ASK
candidates by keyword, and ``data/tau2_ask_label_audit.jsonl`` records the
check of each candidate against its task text. This script scores one or more
tau2 runs on three task subsets built from that file:

* ``all``    -- every keyword candidate;
* ``paper``  -- the subset the paper reports (``in_paper_ask_subset``);
* ``valid``  -- candidates labelled ``valid`` only.

ASK GAR is the share of subset tasks in which the agent emits an ASK-class
message; Acc is the share whose trajectory passes (reward 1); both use the
same detectors as ``compute_tau2_calibration_breakdown.py``.

Usage::

    python scripts/tau2_ask_subset.py \
        --run gpt-5-mini airline $TAU2_DATA_DIR/simulations/tau2_airline_gpt-5-mini_baseline/results.json \
        --run gpt-5-mini retail  $TAU2_DATA_DIR/simulations/tau2_retail_gpt-5-mini_baseline/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from _tau2_common import is_scored_simulation, pass_at_one_hit  # noqa: E402
from compute_tau2_calibration_breakdown import _detect_actions_tau2  # noqa: E402
from bfcl_calibration.analysis.action_classifier import Action  # noqa: E402

SUBSETS = ("all", "paper", "valid")


def ask_subsets(label_audit: Path) -> dict[str, dict[str, list[str]]]:
    """{domain: {subset: [task_id, ...]}} from the released ASK-label audit."""
    subsets: dict[str, dict[str, list[str]]] = {}
    for line in label_audit.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        dom = subsets.setdefault(row["domain"], {name: [] for name in SUBSETS})
        dom["all"].append(row["task_id"])
        if row["in_paper_ask_subset"]:
            dom["paper"].append(row["task_id"])
        if row["label"] == "valid":
            dom["valid"].append(row["task_id"])
    return subsets


def per_task_flags(results_path: Path) -> dict[str, dict[str, bool]]:
    """{task_id: {ask_emit, pass_hit}} over the scored simulations of one run."""
    data = json.loads(results_path.read_text())
    out: dict[str, dict[str, bool]] = {}
    for sim in data.get("simulations") or []:
        if not isinstance(sim, dict) or not is_scored_simulation(sim):
            continue
        emit: set = set()
        for msg in sim.get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                emit |= _detect_actions_tau2(msg)
        out[str(sim.get("task_id"))] = {
            "ask_emit": Action.ASK in emit,
            "pass_hit": pass_at_one_hit(sim.get("reward_info") or {}),
        }
    return out


def score(flags: dict[str, dict[str, bool]], task_ids: list[str]) -> dict:
    present = [t for t in task_ids if t in flags]
    n = len(present)
    if not n:
        return {"n": 0, "gar": None, "acc": None, "delta": None}
    gar = 100.0 * sum(flags[t]["ask_emit"] for t in present) / n
    acc = 100.0 * sum(flags[t]["pass_hit"] for t in present) / n
    return {"n": n, "gar": gar, "acc": acc, "delta": gar - acc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", nargs=3, action="append", required=True,
                    metavar=("MODEL", "DOMAIN", "RESULTS_JSON"),
                    help="one tau2 run: model label, domain (airline|retail), results.json path")
    ap.add_argument("--label-audit", type=Path, default=REPO / "data/tau2_ask_label_audit.jsonl")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    subsets = ask_subsets(args.label_audit)
    rows = []
    for model, domain, path in args.run:
        flags = per_task_flags(Path(path))
        rows.append({"model": model, "domain": domain,
                     **{name: score(flags, subsets[domain][name]) for name in SUBSETS}})

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'model':<20}{'domain':<9}" + "".join(f"{name + ' (n GAR Acc Δ)':>30}" for name in SUBSETS))
    for row in rows:
        cells = []
        for name in SUBSETS:
            s = row[name]
            cells.append(f"{s['n']:>6} {s['gar']:7.1f} {s['acc']:7.1f} {s['delta']:+7.1f}" if s["n"] else f"{'-':>30}")
        print(f"{row['model']:<20}{row['domain']:<9}" + "".join(f"{c:>30}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
