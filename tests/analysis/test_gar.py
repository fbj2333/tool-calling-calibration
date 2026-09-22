from __future__ import annotations

import json

import pytest

from bfcl_calibration.analysis.action_classifier import Action
from bfcl_calibration.analysis.gar import (
    case_actions,
    emission_action,
    gold_action_recall,
    summarize,
)


@pytest.mark.parametrize(
    ("emission", "expected"),
    [
        ('[mkdir(dir_name="tmp")]', Action.TOOL_CALL),  # local-model text call
        ('<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>', Action.TOOL_CALL),
        ('```json\n[{"name": "ls", "arguments": {}}]\n```', Action.TOOL_CALL),  # fenced payload
        ([{"cd": '{"folder": "document"}'}], Action.TOOL_CALL),  # API handler: JSON-string args
        ([{"name": "ls", "arguments": {}}], Action.TOOL_CALL),
        ("Could you tell me which folder?", Action.ASK),
        ("I cannot do that with the available tools.", Action.REFUSE),
        ("All done, the file has been successfully moved.", Action.CONFIRM),
        ("Here is the summary.", Action.OTHER),
        ("", None),
        ("   ", None),
        ([], None),
    ],
)
def test_emission_action(emission, expected):
    assert emission_action(emission) == expected


def test_case_actions_unions_turns_and_ignores_errors():
    result = [
        ['[cd(folder="a")]', "Which file should I move?"],
        [[{"mv": '{"source": "x", "destination": "y"}'}], "Done."],
    ]
    assert case_actions(result) == {Action.TOOL_CALL, Action.ASK, Action.OTHER}
    assert case_actions("Error during inference: timeout") == set()


def test_gold_action_recall_is_case_level_any_turn():
    records = [
        {"result": [['[ls()]'], ["Could you give me the file name?"]]},  # asks in turn 2 -> hit
        {"result": [['[ls()]'], ['[cat(file_name="a")]']]},  # never asks -> miss
        {"result": "Error during inference: boom"},  # counted, never a hit
    ]
    assert gold_action_recall(records, "miss_param") == (1, 3)
    assert gold_action_recall(records, "base") == (2, 3)


def test_summarize_reads_bfcl_layout(tmp_path):
    results = tmp_path / "bfcl_results" / "some-model" / "multi_turn"
    scores = tmp_path / "bfcl_scores" / "some-model" / "multi_turn"
    results.mkdir(parents=True)
    scores.mkdir(parents=True)
    rows = [{"id": f"multi_turn_miss_func_{i}", "result": [["I cannot do that."]] if i < 3 else [['[ls()]']]}
            for i in range(4)]
    (results / "BFCL_v4_multi_turn_miss_func_result.json").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    (scores / "BFCL_v4_multi_turn_miss_func_score.json").write_text(
        json.dumps({"accuracy": 0.5, "correct_count": 2, "total_count": 4}) + "\n")

    summary = summarize(tmp_path / "bfcl_results", tmp_path / "bfcl_scores")

    assert list(summary) == ["miss_func"]
    row = summary["miss_func"]
    assert row["gold_action"] == "REFUSE"
    assert (row["n"], row["gar"], row["acc"], row["gar_minus_acc"]) == (4, 75.0, 50.0, 25.0)
