"""Gold Action Recall (GAR) and accuracy for BFCL multi-turn runs.

Reads the standard BFCL output layout, as written by ``bfcl generate`` /
``bfcl evaluate`` and by ``python -m bfcl_calibration.evaluation.bfcl``::

    <results>/<model>/multi_turn/BFCL_v4_multi_turn_<category>_result.json
    <scores>/<model>/multi_turn/BFCL_v4_multi_turn_<category>_score.json

GAR for a category is the share of its cases in which the model emits the
category's gold action class (``CATEGORY_GOLD_ACTION``) in at least one turn.
Acc is BFCL's own state-graded accuracy, read from the score file. The paper
reports both per category, plus their difference GAR - Acc.

Usage::

    python -m bfcl_calibration.analysis.gar --results runs/my_model/bfcl_results \
        --scores runs/my_model/bfcl_scores
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from bfcl_calibration.analysis.action_classifier import (
    CATEGORY_GOLD_ACTION,
    Action,
    classify_emission,
)
from bfcl_calibration.analysis.emission_normalization import normalize_emission

CATEGORIES = ("base", "long_context", "miss_func", "miss_param")


def _api_call_names(emission: Any) -> list[str]:
    """Tool names in an API-handler emission: ``[{name: args}, ...]`` or ``{name: args}``."""
    calls = emission if isinstance(emission, list) else [emission]
    names: list[str] = []
    for call in calls:
        if not isinstance(call, dict):
            return []
        names.extend(str(name) for name in call)
    return names


def emission_action(emission: Any) -> Action | None:
    """Action class of one saved emission; ``None`` when it carries nothing to classify.

    Local-model handlers save emissions as text; API handlers save tool calls
    as ``[{tool_name: arguments}, ...]`` and text replies as strings.
    """
    normalized = normalize_emission(emission)
    if isinstance(normalized, str):
        return classify_emission(normalized) if normalized.strip() else None
    return Action.TOOL_CALL if _api_call_names(emission) else None


def case_actions(result: Any) -> set[Action]:
    """Union of action classes a case emits across all of its turns."""
    actions: set[Action] = set()
    if not isinstance(result, list):  # e.g. "Error during inference: ..."
        return actions
    for turn in result:
        for emission in turn if isinstance(turn, list) else [turn]:
            action = emission_action(emission)
            if action is not None:
                actions.add(action)
    return actions


def gold_action_recall(records: Iterable[dict[str, Any]], category: str) -> tuple[int, int]:
    """(cases that emit the gold action, cases) for one category."""
    gold = CATEGORY_GOLD_ACTION[category]
    hits = n = 0
    for record in records:
        n += 1
        hits += gold in case_actions(record.get("result"))
    return hits, n


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _find(root: Path, category: str, kind: str) -> Path | None:
    hits = sorted(root.glob(f"**/BFCL_v4_multi_turn_{category}_{kind}.json"))
    if len(hits) > 1:
        raise SystemExit(f"{root}: more than one {kind} file for {category}: {hits}")
    return hits[0] if hits else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _score_header(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.loads(fh.readline())


def summarize(results: Path, scores: Path | None = None) -> dict[str, dict[str, Any]]:
    """Per-category GAR (and Acc, when score files are given) for one run."""
    summary: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        result_file = _find(results, category, "result")
        if result_file is None:
            continue
        hits, n = gold_action_recall(_read_jsonl(result_file), category)
        low, high = wilson_interval(hits, n)
        row: dict[str, Any] = {
            "gold_action": CATEGORY_GOLD_ACTION[category].name,
            "n": n,
            "gar": 100 * hits / n if n else None,
            "gar_ci95": [100 * low, 100 * high],
        }
        score_file = _find(scores, category, "score") if scores else None
        if score_file is not None:
            header = _score_header(score_file)
            row["acc"] = 100 * header["accuracy"]
            row["gar_minus_acc"] = row["gar"] - row["acc"]
        summary[category] = row
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, required=True, help="BFCL result directory")
    parser.add_argument("--scores", type=Path, help="BFCL score directory (adds Acc and GAR - Acc)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    summary = summarize(args.results, args.scores)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    print(f"{'category':<14}{'gold':<11}{'n':>5}{'GAR':>8}{'Acc':>8}{'GAR-Acc':>9}")
    for category, row in summary.items():
        acc = f"{row['acc']:.1f}" if "acc" in row else "-"
        gap = f"{row['gar_minus_acc']:+.1f}" if "acc" in row else "-"
        print(f"{category:<14}{row['gold_action']:<11}{row['n']:>5}{row['gar']:>8.1f}{acc:>8}{gap:>9}")
    if summary and all("acc" in row for row in summary.values()):
        overall = sum(row["acc"] for row in summary.values()) / len(summary)
        print(f"{'overall Acc':<30}{overall:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
