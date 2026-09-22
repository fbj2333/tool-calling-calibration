from __future__ import annotations

from types import SimpleNamespace

import pytest

from bfcl_calibration.evaluation import bfcl
from bfcl_calibration.evaluation.bfcl._categories import _resolve_categories


def test_build_subset_mapping_caps_entries_per_category() -> None:
    handles = SimpleNamespace(
        load_dataset_entry=lambda category: [
            {"id": f"{category}_0"},
            {"id": f"{category}_1"},
            {"id": f"{category}_2"},
        ]
    )

    mapping = bfcl._build_subset_mapping(
        handles,
        categories=["multi_turn_base", "multi_turn_miss_func"],
        max_samples=2,
        run_ids_mapping=None,
    )

    assert mapping == {
        "multi_turn_base": ["multi_turn_base_0", "multi_turn_base_1"],
        "multi_turn_miss_func": ["multi_turn_miss_func_0", "multi_turn_miss_func_1"],
    }


def test_resolve_categories_defaults_to_all_four_multi_turn_categories() -> None:
    assert _resolve_categories(None, None) == list(bfcl.MULTI_TURN_CATEGORIES)
    assert _resolve_categories(["multi_turn_miss_param,multi_turn_base"], None) == [
        "multi_turn_miss_param",
        "multi_turn_base",
    ]
    assert _resolve_categories(None, {"multi_turn_miss_func": ["multi_turn_miss_func_0"]}) == [
        "multi_turn_miss_func"
    ]


def test_resolve_categories_rejects_non_multi_turn_category() -> None:
    with pytest.raises(ValueError, match="simple_python"):
        _resolve_categories(["simple_python"], None)
