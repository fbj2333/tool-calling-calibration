#!/usr/bin/env python3
"""Audit tau2-bench reward_basis + DB-unchanged/abstain-correct distribution.

Primary-source verification for the paper's cross-benchmark positioning: it
quantifies, per domain, (a) the reward_basis composition (which scoring axes
each task uses, in particular whether the LLM NL-assertion judge is in the
reward path) and (b) how many tasks have a unique correct outcome that leaves
the database unchanged (no reference WRITE action) -- the ceiling for any
"models over-call even when the correct action is to abstain" statement.

Run against an official sierra-research/tau2-bench checkout pinned at the
commit the paper used:

    git clone https://github.com/sierra-research/tau2-bench.git
    git -C tau2-bench checkout a03b7910bc968f706306e16017a9cb650caf7af2
    python scripts/tau2_reward_basis_audit.py --tau2-root /path/to/tau2-bench

A WRITE tool is a domain toolkit method decorated @is_tool(ToolType.WRITE);
the set is parsed from source so the result tracks the pinned commit exactly.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import Counter
from pathlib import Path

EXPECTED_COMMIT = "a03b7910bc968f706306e16017a9cb650caf7af2"
EXPECTED_N = {"retail": 114, "airline": 50}


def write_tools(root: Path, domain: str) -> set[str]:
    src = (root / f"src/tau2/domains/{domain}/tools.py").read_text()
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and getattr(dec.func, "id", "") == "is_tool"
                and dec.args
                and isinstance(dec.args[0], ast.Attribute)
                and dec.args[0].attr == "WRITE"
            ):
                out.add(node.name)
    return out


def audit(root: Path, domain: str) -> dict:
    wt = write_tools(root, domain)
    tasks = json.loads((root / f"data/tau2/domains/{domain}/tasks.json").read_text())
    rb_combo: Counter = Counter()
    rb_flag: Counter = Counter()
    zero_actions, no_write, no_write_no_nl = [], [], []
    for t in tasks:
        ec = t.get("evaluation_criteria") or {}
        rb = ec.get("reward_basis") or []
        rb_combo["+".join(sorted(rb)) or "(none)"] += 1
        for f in rb:
            rb_flag[f] += 1
        actions = ec.get("actions") or []
        names = [a.get("name") for a in actions]
        if not actions:
            zero_actions.append(t.get("id"))
        if not any(nm in wt for nm in names):
            no_write.append(t.get("id"))
            if "NL_ASSERTION" not in rb:
                no_write_no_nl.append(t.get("id"))
    return {
        "domain": domain,
        "n": len(tasks),
        "write_tools": sorted(wt),
        "reward_basis_combo": dict(rb_combo),
        "reward_basis_flag": dict(rb_flag),
        "zero_action_ids": zero_actions,
        "db_unchanged_correct_ids": no_write,
        "db_unchanged_correct_no_nl_judge_ids": no_write_no_nl,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2-root", required=True, type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    root = args.tau2_root.resolve()

    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        head = "(not a git checkout)"
    if head != EXPECTED_COMMIT:
        print(f"WARNING: HEAD={head} != pinned {EXPECTED_COMMIT}; results may not match the harness.")

    results = []
    for dom, n in EXPECTED_N.items():
        r = audit(root, dom)
        if r["n"] != n:
            print(f"WARNING: {dom} expected {n} tasks, found {r['n']}")
        results.append(r)
        print(f"\n===== {dom.upper()} (n={r['n']}) =====")
        print(f"WRITE tools ({len(r['write_tools'])}): {r['write_tools']}")
        print(f"reward_basis combinations: {r['reward_basis_combo']}")
        print(f"reward_basis flag presence: {r['reward_basis_flag']}")
        print(f"zero reference actions (pure abstain): "
              f"{len(r['zero_action_ids'])} -> {r['zero_action_ids']}")
        print(f"DB-unchanged-correct (no reference WRITE action): "
              f"{len(r['db_unchanged_correct_ids'])}")
        print(f"  ...of which scored WITHOUT the NL-assertion LLM judge: "
              f"{len(r['db_unchanged_correct_no_nl_judge_ids'])}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"head": head, "results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
