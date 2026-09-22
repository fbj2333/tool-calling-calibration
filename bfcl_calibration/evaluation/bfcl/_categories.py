"""BFCL multi-turn categories and the run-subset helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

MULTI_TURN_CATEGORIES: tuple[str, ...] = (
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
)

_BFCL_RUN_IDS_FILENAME = "test_case_ids_to_generate.json"


def _split_cli_values(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _resolve_categories(
    raw_categories: Sequence[str] | None,
    run_ids_mapping: dict[str, list[str]] | None,
) -> list[str]:
    """Categories to run: ``--categories`` if given, else those of the run-ids file, else all four."""
    requested = _split_cli_values(raw_categories) or list(run_ids_mapping or {})
    if not requested:
        return list(MULTI_TURN_CATEGORIES)
    categories = list(dict.fromkeys(requested))
    unknown = [category for category in categories if category not in MULTI_TURN_CATEGORIES]
    if unknown:
        raise ValueError(
            f"Unsupported BFCL categories {unknown}; expected a subset of {list(MULTI_TURN_CATEGORIES)}."
        )
    return categories


def _load_run_ids_mapping(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--run-ids-file must contain a JSON object mapping category -> list[test_id].")

    mapping: dict[str, list[str]] = {}
    for category, ids in payload.items():
        # Category names are validated by _resolve_categories.
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError(
                f"Run-ids entry for '{category}' must be a list of string test ids."
            )
        if ids:
            mapping[category] = list(ids)
    return mapping


def _build_subset_mapping(
    handles: Any,
    *,
    categories: list[str],
    max_samples: int | None,
    run_ids_mapping: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    if run_ids_mapping is not None:
        filtered = {
            category: run_ids_mapping[category]
            for category in categories
            if category in run_ids_mapping
        }
        missing = [category for category in categories if category not in filtered]
        if missing:
            raise ValueError(
                "run-ids file does not define ids for selected categories: "
                + ", ".join(missing)
            )
        return filtered

    if max_samples is None:
        return None
    if max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer.")

    return {
        category: [
            str(entry["id"])
            for entry in handles.load_dataset_entry(category)[:max_samples]
        ]
        for category in categories
    }


def _write_subset_mapping(project_root: Path, mapping: dict[str, list[str]]) -> Path:
    output_path = project_root / _BFCL_RUN_IDS_FILENAME
    output_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _sanitize_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in value).strip("-") or "run"
