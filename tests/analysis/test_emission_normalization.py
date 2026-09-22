from __future__ import annotations

from bfcl_calibration.analysis.emission_normalization import normalize_emission


def test_canonical_strings_are_noop() -> None:
    examples = [
        '[{"name":"mkdir","arguments":{"dir_name":"tmp"}}]',
        '[mkdir(dir_name="tmp")]',
        '<tool_call>\n{"name":"mkdir","arguments":{"dir_name":"tmp"}}\n</tool_call>',
    ]

    for example in examples:
        assert normalize_emission(example) == example


def test_text_and_degenerate_strings_are_noop() -> None:
    examples = [
        "Could you provide the destination folder?",
        "Assistant:",
        "",
        'Please run this:\n```json\n{"name":"mkdir","arguments":{"dir_name":"tmp"}}\n```',
    ]

    for example in examples:
        assert normalize_emission(example) == example


def test_python_repr_name_arguments_string_becomes_canonical_json() -> None:
    emission = "```\n[{'name': 'mkdir', 'arguments': {'dir_name': 'tmp'}}]\n```"

    assert normalize_emission(emission) == '[{"name":"mkdir","arguments":{"dir_name":"tmp"}}]'


def test_granite_shorthand_list_becomes_canonical_json() -> None:
    emission = [{"mkdir": {"dir_name": "tmp"}}, {"mv": {"source": "a", "destination": "tmp"}}]

    assert normalize_emission(emission) == (
        '[{"name":"mkdir","arguments":{"dir_name":"tmp"}},'
        '{"name":"mv","arguments":{"source":"a","destination":"tmp"}}]'
    )


def test_empty_structures_are_noop() -> None:
    emission: list[object] = []

    assert normalize_emission(emission) is emission
