from bfcl_calibration.evaluation.bfcl._state import (
    _parse_execution_strings,
    _response_data_tool_calls,
)


def test_parse_execution_string_single_call():
    assert _parse_execution_strings(["mkdir(dir_name='temp')"]) == [
        {"name": "mkdir", "arguments": {"dir_name": "temp"}}
    ]


def test_parse_execution_strings_multiple_calls_and_empty_args():
    assert _parse_execution_strings(["cd(folder='document')", "ls()"]) == [
        {"name": "cd", "arguments": {"folder": "document"}},
        {"name": "ls", "arguments": {}},
    ]


def test_parse_execution_string_literal_values():
    assert _parse_execution_strings(
        ["lockDoors(unlock=True, door=['driver', 'passenger'], meta={'ok': 1})"]
    ) == [
        {
            "name": "lockDoors",
            "arguments": {
                "unlock": True,
                "door": ["driver", "passenger"],
                "meta": {"ok": 1},
            },
        }
    ]


def test_response_data_tool_calls_falls_back_without_decoded_field():
    assert _response_data_tool_calls(
        {
            "model_responses": '<tool_call>\n{"name": "mkdir", "arguments": {"dir_name": "temp"}}\n</tool_call>'
        }
    ) == [{"name": "mkdir", "arguments": {"dir_name": "temp"}}]


def test_response_data_tool_calls_falls_back_with_empty_decoded_list():
    assert _response_data_tool_calls(
        {
            "model_responses_decoded": [],
            "model_responses": '<tool_call>\n{"name": "ls", "arguments": {"a": true}}\n</tool_call>',
        }
    ) == [{"name": "ls", "arguments": {"a": True}}]


def test_parse_execution_strings_skips_invalid_entries():
    assert _parse_execution_strings(["not a call", "broken(", "foo('positional')"]) == []


def test_hammer_decoded_execution_strings_parse():
    assert _response_data_tool_calls(
        {
            "model_responses": "```not parsed by CRI raw branches```",
            "model_responses_decoded": [
                "lockDoors(unlock=True, door=['driver', 'passenger', 'rear_left', 'rear_right'])",
                "setHeadlights(mode='on')",
            ],
        }
    ) == [
        {
            "name": "lockDoors",
            "arguments": {
                "unlock": True,
                "door": ["driver", "passenger", "rear_left", "rear_right"],
            },
        },
        {"name": "setHeadlights", "arguments": {"mode": "on"}},
    ]


def test_toolace_decoded_execution_string_parse():
    assert _response_data_tool_calls(
        {
            "model_responses": '[cd(folder="document")]',
            "model_responses_decoded": ["cd(folder='document')"],
        }
    ) == [{"name": "cd", "arguments": {"folder": "document"}}]
