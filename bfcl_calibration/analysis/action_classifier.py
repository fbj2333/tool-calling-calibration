"""Action classifier for BFCL multi-turn model emissions.

Each model emission inside a BFCL turn is classified into one of:
  - TOOL_CALL: parsed function-call list, e.g. "[mv(...), cp(...)]"
  - ASK: text emission asking the user (question mark / asking phrase)
  - REFUSE: text emission stating an action is impossible / unauthorized
  - CONFIRM: text emission asserting task done / completed
  - OTHER: text emission that does not match any of the above

GAR (``bfcl_calibration.analysis.gar``) takes the union of the classes a case
emits across all of its turns (``gar.case_actions``) and counts the cases
whose union contains the category's gold action.

The ASK phrase table matches the one the SRI probe uses at inference time
(``bfcl_calibration/evaluation/bfcl/_sri.py``), so analysis and intervention
read asking the same way.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable


class Action(str, Enum):
    TOOL_CALL = "tool_call"
    ASK = "ask"
    REFUSE = "refuse"
    CONFIRM = "confirm"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Phrase tables (ASK kept consistent with the SRI probe's asking heuristic).
# ---------------------------------------------------------------------------

ASK_PHRASES: tuple[str, ...] = (
    "could you",
    "please provide",
    "please specify",
    "please tell",
    "please confirm",
    "i need to know",
    "i need the",
    "what is the",
    "what's the",
    "kindly provide",
    "would you provide",
    "i'll need",
    "could you confirm",
    "could you let me know",
    "could you tell",
    "please clarify",
    "please share",
    "please indicate",
)

REFUSE_PHRASES: tuple[str, ...] = (
    "cannot",
    "can't",
    "unable to",
    "not able to",
    "no such function",
    "no function",
    "function is not available",
    "no such tool",
    "not available in",
    "is not provided",
    "is not supported",
    "do not have access",
    "don't have access",
    "no available function",
    "no matching function",
    "not authorized",
    "not permitted",
    "not possible",
    "i don't have a tool",
    "i do not have a tool",
    "no tool available",
)

CONFIRM_PHRASES: tuple[str, ...] = (
    "task complete",
    "all tasks",
    "all set",
    "all done",
    "fully done",
    "wrapped up",
    "completed successfully",
    "successfully completed",
    "successfully moved",
    "successfully created",
    "successfully deleted",
    "has been successfully",
    "have been successfully",
    "no further actions",
    "no further action",
    "no additional actions",
    "everything is",
    "everything's",
    "i have completed",
    "i've completed",
    "i have finished",
    "i've finished",
    "all the requested",
)

# Local-model tool calls appear in BFCL result files as text in one of three
# forms, depending on the handler:
#   1. Python call list:        "[mv(source=\"a\", destination=\"b\")]"
#   2. <tool_call> XML:         "<tool_call>\n{\"name\": ..., ...}\n</tool_call>"
#                               (the gpt-oss handler writes this form)
#   3. JSON name/arguments list: "[{\"name\": \"mv\", \"arguments\": {...}}, ...]"
# API-model tool calls are structured objects; ``emission_normalization``
# converts them before classification.
_TOOL_CALL_PY_RE = re.compile(r"^\s*\[\s*[A-Za-z_][A-Za-z0-9_]*\s*\(")
_TOOL_CALL_XML_RE = re.compile(r"<tool_call\b", flags=re.IGNORECASE)
_TOOL_CALL_JSON_RE = re.compile(
    r"^\s*\[\s*\{\s*[\"']name[\"']\s*:", flags=re.MULTILINE | re.DOTALL
)
_TOOL_CALL_SINGLE_JSON_RE = re.compile(
    r"^\s*\{\s*[\"']name[\"']\s*:\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*,"
    r".*[\"'](?:arguments|parameters)[\"']\s*:",
    flags=re.MULTILINE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Single-emission classification
# ---------------------------------------------------------------------------


def is_tool_call_emission(text: str) -> bool:
    if not text:
        return False
    if _TOOL_CALL_PY_RE.match(text):
        return True
    if _TOOL_CALL_XML_RE.search(text):
        return True
    if _TOOL_CALL_JSON_RE.match(text):
        return True
    if _TOOL_CALL_SINGLE_JSON_RE.match(text):
        return True
    return False


def _contains_any(lower_text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in lower_text for phrase in phrases)


# Typographic-quote normalization: closed-source / RLHF-tuned models
# (e.g., gpt-5.x, Claude) frequently emit U+2019 RIGHT SINGLE QUOTATION MARK
# instead of ASCII apostrophe inside contractions like "can't" / "don't" /
# "isn't". Without this, REFUSE_PHRASES fails to match. Also normalize
# curly double quotes for completeness.
_QUOTE_NORMALIZE = str.maketrans({
    "‘": "'",  # ‘ left single
    "’": "'",  # ’ right single
    "‚": "'",  # ‚ single low-9
    "‛": "'",  # ‛ single high-reversed-9
    "“": '"',  # “ left double
    "”": '"',  # ” right double
    "„": '"',  # „ double low-9
    "‟": '"',  # ‟ double high-reversed-9
    "´": "'",  # ´ acute accent (occasionally used as apostrophe)
    "ʼ": "'",  # ʼ modifier letter apostrophe
})


def _normalize(text: str) -> str:
    return text.translate(_QUOTE_NORMALIZE)


def classify_emission(text: str | None) -> Action:
    """Classify a single model emission string.

    Priority: TOOL_CALL > REFUSE > ASK > CONFIRM > OTHER.

    REFUSE is checked before ASK because some refusals contain question
    marks ("Are you sure? I cannot do that."); we want REFUSE to win.
    """
    if text is None:
        return Action.OTHER
    if is_tool_call_emission(text):
        return Action.TOOL_CALL
    normalized = _normalize(text)
    lower = normalized.lower()
    if _contains_any(lower, REFUSE_PHRASES):
        return Action.REFUSE
    if "?" in normalized or _contains_any(lower, ASK_PHRASES):
        return Action.ASK
    if _contains_any(lower, CONFIRM_PHRASES):
        return Action.CONFIRM
    return Action.OTHER


# ---------------------------------------------------------------------------
# BFCL category -> gold action mapping
# ---------------------------------------------------------------------------


CATEGORY_GOLD_ACTION: dict[str, Action] = {
    "base": Action.TOOL_CALL,
    "long_context": Action.TOOL_CALL,
    "miss_param": Action.ASK,
    "miss_func": Action.REFUSE,
}


def category_from_case_id(case_id: str) -> str | None:
    """Extract BFCL category prefix from a case id like
    'multi_turn_miss_param_42'.
    """
    if not case_id:
        return None
    for cat in ("miss_param", "miss_func", "long_context", "base"):
        if f"multi_turn_{cat}_" in case_id:
            return cat
    return None
