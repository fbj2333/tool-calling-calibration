#!/usr/bin/env python3
"""Tab. 3: confirmation before database writes on tau2-bench.

Reads ``data/tau2_confirm_judgments.jsonl`` (one judged write per line). A write
counts when the judge returned a verdict (``confirmed`` is true or false). Per
model and domain:

* GAR -- share of counted writes preceded by an explicit user confirmation;
* Acc -- share of the same writes whose trajectory passes tau2-bench's reward;
* n   -- counted writes. Models that never reach a write have no rate.

Rows: ``--models`` if given; for the released file, the rows of Tab. 3 (which
include two models that never reach a write); otherwise every model in the file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RELEASED = REPO / "data/tau2_confirm_judgments.jsonl"
TAB3_ROWS = ["Qwen3-8B", "Qwen3-14B", "gpt-oss-20b", "xLAM-2-8b", "Hammer-2.1-7B", "ToolACE-2-8B",
         "gpt-5", "gpt-5.4", "gpt-5-mini", "Gemini 3 Pro", "Doubao-Seed-1.8"]
DOMAINS = ("airline", "retail")


def cells(records: list[dict], models: list[str]) -> dict[tuple[str, str], tuple[int, float | None, float | None]]:
    out = {}
    for model in models:
        for domain in DOMAINS:
            counted = [r for r in records if r["model"] == model and r["domain"] == domain
                       and isinstance(r["confirmed"], bool)]
            n = len(counted)
            if n == 0:
                out[(model, domain)] = (0, None, None)
                continue
            gar = 100 * sum(r["confirmed"] for r in counted) / n
            acc = 100 * sum(bool(r["passed"]) for r in counted) / n
            out[(model, domain)] = (n, gar, acc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--judgments", type=Path, default=RELEASED)
    ap.add_argument("--models", nargs="+", help="rows to print, in order")
    args = ap.parse_args()
    records = [json.loads(line) for line in args.judgments.read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.models:
        models = args.models
    elif args.judgments.resolve() == RELEASED.resolve():
        models = TAB3_ROWS
    else:
        models = list(dict.fromkeys(r["model"] for r in records))
    table = cells(records, models)
    head = "".join(f"{d + ' GAR':>13}{'Acc':>7}{'Δ':>7}{'n':>5}" for d in DOMAINS)
    print(f"{'model':<18}{head}")
    for model in models:
        row = ""
        for domain in DOMAINS:
            n, gar, acc = table[(model, domain)]
            row += f"{'--':>13}{'--':>7}{'--':>7}{0:>5}" if n == 0 else f"{gar:>13.1f}{acc:>7.1f}{gar - acc:>+7.1f}{n:>5}"
        print(f"{model:<18}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
