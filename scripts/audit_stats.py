#!/usr/bin/env python3
"""App. F: agreement and mechanism rates for the 200-case mechanism audit.

Reads ``data/audit_labels.jsonl`` (``split == "main"``) and prints

1. per-label Cohen's kappa and raw agreement between the two raters before
   adjudication, over cases both raters labelled (not ``n.a.``), plus the
   macro-mean and minimum over the five primary labels;
2. post-adjudication rates with Wilson 95% intervals (B3 over all
   ``miss_func`` cases and over ``miss_func`` cases with B1a = 0);
3. per-stratum agreement on the primary labels;
4. the post-adjudication B7 distribution.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABELS = ["B1a", "B1b", "B3", "B4", "B5", "B6", "B7"]
PRIMARY = ["B1a", "B1b", "B3", "B4", "B5"]
NA = "n.a."


def cohen_kappa(pairs: list[tuple]) -> float:
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    count_a, count_b = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / denom, (centre + half) / denom


def rater_pairs(cases: list[dict], label: str) -> list[tuple]:
    return [(c["rater_a"][label]["value"], c["rater_b"][label]["value"]) for c in cases
            if c["rater_a"][label]["value"] != NA and c["rater_b"][label]["value"] != NA]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", type=Path, default=REPO / "data/audit_labels.jsonl")
    args = ap.parse_args()
    cases = [json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    main_cases = [c for c in cases if c["split"] == "main"]

    print(f"Pre-adjudication agreement (n = {len(main_cases)} cases)")
    print(f"{'label':<6}{'n':>6}{'agreement':>11}{'kappa':>9}")
    primary_kappas = []
    for label in LABELS:
        pairs = rater_pairs(main_cases, label)
        kappa = cohen_kappa(pairs)
        if label in PRIMARY:
            primary_kappas.append(kappa)
        agreement = sum(a == b for a, b in pairs) / len(pairs)
        print(f"{label:<6}{len(pairs):>6}{agreement:>11.3f}{kappa:>9.3f}")
    print(f"primary macro-mean kappa {sum(primary_kappas) / len(primary_kappas):.4f}, "
          f"min {min(primary_kappas):.4f}\n")

    print("Post-adjudication rates")
    print(f"{'label':<8}{'k/n':>9}{'rate':>8}{'Wilson 95%':>18}")
    def rate_row(name: str, values: list) -> None:
        k, n = sum(v == 1 for v in values), len(values)
        low, high = wilson(k, n)
        print(f"{name:<8}{f'{k}/{n}':>9}{k / n:>8.3f}{f'[{low:.3f}, {high:.3f}]':>18}")
    for label in ["B1a", "B1b"]:
        rate_row(label, [c["final"][label] for c in main_cases])
    miss_func = [c for c in main_cases if c["category"] == "miss_func"]
    rate_row("B3 D1", [c["final"]["B3"] for c in miss_func])
    rate_row("B3 D2", [c["final"]["B3"] for c in miss_func if c["final"]["B1a"] == 0])
    for label in ["B4", "B5", "B6"]:
        rate_row(label, [c["final"][label] for c in main_cases if c["final"][label] != NA])

    print("\nPre-adjudication agreement per stratum (primary labels)")
    print(f"{'stratum':<9}{'n':>4}" + "".join(f"{label:>7}" for label in PRIMARY))
    for stratum in sorted({c["stratum"] for c in main_cases}):
        group = [c for c in main_cases if c["stratum"] == stratum]
        cells = []
        for label in PRIMARY:
            pairs = [(c["rater_a"][label]["value"], c["rater_b"][label]["value"]) for c in group]
            cells.append(sum(a == b for a, b in pairs) / len(pairs))
        print(f"{stratum:<9}{len(group):>4}" + "".join(f"{v:>7.2f}" for v in cells))

    b7 = Counter(c["final"]["B7"] for c in main_cases if c["final"]["B7"] != NA)
    print(f"\nPost-adjudication B7 (n = {sum(b7.values())}): " + ", ".join(f"{k} {v}" for k, v in b7.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
