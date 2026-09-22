"""Unit tests for bfcl_calibration.analysis.action_classifier."""

from __future__ import annotations

import unittest

from bfcl_calibration.analysis.action_classifier import (
    Action,
    CATEGORY_GOLD_ACTION,
    category_from_case_id,
    classify_emission,
    is_tool_call_emission,
)


class TestToolCallDetection(unittest.TestCase):
    def test_python_list_simple_call(self):
        # prompt-mode canonical form
        assert is_tool_call_emission('[mv(source="a", destination="b")]')

    def test_python_list_multi_call(self):
        assert is_tool_call_emission(
            '[mkdir(dir_name="temp"), mv(source="x", destination="temp")]'
        )

    def test_python_list_call_with_leading_whitespace(self):
        assert is_tool_call_emission('  [cd(folder="document")]')

    def test_xml_tool_call_single(self):
        # FC-mode default canonical form (Qwen / xLAM / gpt-oss)
        assert is_tool_call_emission(
            '<tool_call>\n{"name": "mv", "arguments": {"source": "a", "destination": "b"}}\n</tool_call>'
        )

    def test_xml_tool_call_multi(self):
        text = (
            '<tool_call>\n{"name": "mkdir", "arguments": {"dir_name": "temp"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "mv", "arguments": {"source": "x", "destination": "temp"}}\n</tool_call>'
        )
        assert is_tool_call_emission(text)

    def test_xml_tool_call_uppercase(self):
        # case-insensitive XML tag
        assert is_tool_call_emission('<TOOL_CALL>{"name": "x", "arguments": {}}</TOOL_CALL>')

    def test_json_name_arguments_list(self):
        # FC-mode JSON canonical form when raw lacks XML
        assert is_tool_call_emission(
            '[{"name": "mv", "arguments": {"source": "a", "destination": "b"}}]'
        )

    def test_json_name_arguments_multi(self):
        assert is_tool_call_emission(
            '[{"name": "mkdir", "arguments": {"dir_name": "temp"}}, '
            '{"name": "mv", "arguments": {"source": "x", "destination": "y"}}]'
        )

    def test_json_single_quote_name(self):
        assert is_tool_call_emission("[{'name': 'mv', 'arguments': {}}]")

    def test_text_starting_with_bracket_but_not_call(self):
        assert not is_tool_call_emission("[Note: this is just a comment]")

    def test_plain_text(self):
        assert not is_tool_call_emission("The file has been moved.")

    def test_text_mentioning_tool_call_word(self):
        # word "tool_call" without XML-tag wrapping should NOT match
        assert not is_tool_call_emission(
            "I will issue a tool_call to mv shortly."
        )

    def test_empty(self):
        assert not is_tool_call_emission("")


class TestSingleEmissionClassify(unittest.TestCase):
    def test_tool_call_takes_priority(self):
        assert classify_emission('[mv(source="a", destination="b")]') == Action.TOOL_CALL

    def test_clear_ask_with_question_mark(self):
        assert classify_emission("Could you tell me which folder?") == Action.ASK

    def test_clear_ask_via_phrase_no_question_mark(self):
        assert classify_emission("I need the folder name to proceed.") == Action.ASK

    def test_clear_refuse(self):
        assert (
            classify_emission("I cannot complete this task as the function is not available.")
            == Action.REFUSE
        )

    def test_refuse_wins_over_ask_when_both_present(self):
        # A refuse with a trailing question mark should still classify as refuse
        text = "I cannot perform that action. Is there something else?"
        assert classify_emission(text) == Action.REFUSE

    def test_clear_confirm(self):
        assert (
            classify_emission(
                "The file has been successfully moved. No further actions are required."
            )
            == Action.CONFIRM
        )

    def test_other_text(self):
        assert classify_emission("Hello world.") == Action.OTHER

    def test_none_is_other(self):
        assert classify_emission(None) == Action.OTHER

    def test_empty_is_other(self):
        assert classify_emission("") == Action.OTHER


class TestCategoryGold(unittest.TestCase):
    def test_category_extraction(self):
        assert category_from_case_id("multi_turn_base_0") == "base"
        assert category_from_case_id("multi_turn_long_context_5") == "long_context"
        assert category_from_case_id("multi_turn_miss_param_12") == "miss_param"
        assert category_from_case_id("multi_turn_miss_func_42") == "miss_func"

    def test_category_extraction_unknown(self):
        assert category_from_case_id("unrelated_id") is None
        assert category_from_case_id("") is None

    def test_gold_actions_present(self):
        # Sanity: every BFCL category has a gold action mapping
        for cat in ("base", "long_context", "miss_param", "miss_func"):
            assert cat in CATEGORY_GOLD_ACTION

    def test_gold_action_values(self):
        # Pre-registered (paper Section 3): base/long_context expect tool calls,
        # miss_param expects an ask, miss_func expects a refuse.
        assert CATEGORY_GOLD_ACTION["base"] == Action.TOOL_CALL
        assert CATEGORY_GOLD_ACTION["long_context"] == Action.TOOL_CALL
        assert CATEGORY_GOLD_ACTION["miss_param"] == Action.ASK
        assert CATEGORY_GOLD_ACTION["miss_func"] == Action.REFUSE
