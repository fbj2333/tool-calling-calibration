#!/usr/bin/env python3
"""Per-family per-domain capability/calibration breakdown on tau2-bench.

Applies the paper's diagnostic framework to tau2 baseline simulation outputs.

Capability and calibration decompose over a **four-class** action
space ``{tool_call, ask, refuse, confirm}``. The canonical ``Action`` enum
in :mod:`bfcl_calibration.analysis.action_classifier` has 5 values
(``TOOL_CALL/ASK/REFUSE/CONFIRM/OTHER``); ``OTHER`` is a priority-chain
fallback for emissions that signal no decision and is not part of the
framework. CONFIRM has no task-level gold (Tab. 3 scores it per database
write, see ``judge_tau2_confirm.py``), so calibration here is over **three**
classes (TOOL_CALL/ASK/REFUSE) while capability is over **four**.

Output per (family, domain, method) row contains 8 measurements:

* ``pass@1``                                (final-state task success)
* ``capability_tool_call/ask/refuse/confirm`` (4 classes)
* ``calibration_tool_call/ask/refuse``        (3 gold-anchored classes)

Calibration semantics:

* ``calibration_refuse`` = % of REFUSE-required tasks (BFCL miss_func
  analog) where the model emitted 0 WRITE tool calls (restraint).
* ``calibration_ask`` = % of ASK-required tasks (BFCL miss_param analog)
  where the model emitted at least one ASK-class emission.
* ``calibration_tool_call`` = action-GAR
  (no failed reference action, ``failed_action_count == 0``) on the
  TOOL_CALL-required subset (tasks with at least one reference action).

Input: each ``--runs`` directory holds tau2 run directories named
``tau2_<domain>_<model>_<method>`` (method: ``baseline``, ``sri_lite``, or
``ablation_no_state``), each with its ``results.json``; ``tau2 run --save-to``
writes them under ``$TAU2_DATA_DIR/simulations``. The printed table shows GAR;
``--output-json`` / ``--output-csv`` add Acc and GAR - Acc per class.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tau2_common import (  # noqa: E402
    failed_action_count,
    is_scored_simulation,
    pass_at_one_hit,
)
from tau2_reward_basis_audit import write_tools  # noqa: E402

# Canonical 4-class action enum + BFCL phrase tables / structural
# helpers. We re-use REFUSE_PHRASES, ASK_PHRASES as lexicon and
# is_tool_call_emission as the canonical text-format tool-call detector
# (xLAM, Hammer, and other handlers emit tool calls in ``content`` as
# JSON / XML / Python-call-list when their handler does not populate
# the structured ``tool_calls`` field). We do NOT re-use BFCL's cascade
# classifier (``classify_emission``): its priority order REFUSE > ASK
# collapses any "I can't do X, please send Y" message into REFUSE
# alone, undercounting ASK on frontier models that routinely combine
# the two. τ² needs an independent multi-label detector (see
# :func:`_detect_actions_tau2` below).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bfcl_calibration.analysis.action_classifier import (  # noqa: E402
    Action,
    REFUSE_PHRASES,
    ASK_PHRASES,
    is_tool_call_emission,
    _normalize as _normalize_typographic_quotes,
)

DOMAINS = ("airline", "retail")

# Capability measured over the four framework classes.
# OTHER intentionally excluded — it is a priority-chain fallback, not a
# framework class.
CAPABILITY_CLASSES: tuple[Action, ...] = (
    Action.TOOL_CALL,
    Action.ASK,
    Action.REFUSE,
    Action.CONFIRM,
)

# tau2_<domain>_<model>_<method>, optionally followed by _<YYYY-MM-DD><suffix>.
_RUN_DIR = re.compile(
    r"^tau2_(?P<domain>airline|retail)_"
    r"(?P<family>.+?)_"
    r"(?P<method>ablation_no_state|sri_lite|baseline)"
    r"(?:_\d{4}-\d{2}-\d{2}.*)?$"
)


@dataclass(frozen=True)
class RunInfo:
    domain: str
    family: str
    method: str
    path: Path


def parse_run_dir(path: Path) -> RunInfo | None:
    m = _RUN_DIR.match(path.parent.name)
    if m is None:
        return None
    return RunInfo(domain=m["domain"], family=m["family"], method=m["method"], path=path)


# Some vLLM tool-call parsers (e.g. ``hermes``, used when serving
# Hammer-2.1-7b-fc) return tool-call JSON inside markdown code fences in the
# assistant ``content`` and leave the structured ``tool_calls`` field empty.
# ``classify_emission`` anchors on a leading ``[``/``{``, so fences are
# stripped here before classification.
_MARKDOWN_FENCE_RE = re.compile(r"```(?:[a-zA-Z]*\n)?(.*?)```", re.DOTALL)


def _unfence_markdown(text: str | None) -> str | None:
    if text is None:
        return None
    return _MARKDOWN_FENCE_RE.sub(lambda m: m.group(1).strip(), text).strip()


# τ²-bench action detection. Three choices differ from the BFCL classifier:
#
# (1) Multi-label rather than a priority cascade: each class is detected
# independently, so one message can carry both REFUSE and ASK. Messages such
# as "I can't do X without Y, please provide Y" are common in customer
# service; a cascade (REFUSE > ASK > ...) would record only REFUSE.
#
# (2) ASK accepts any ASK_PHRASE imperative or "?"-question whose sentence is
# not a greeting or closing ("Hi! How can I help today?"), so open-ended
# clarifying questions count and routine pleasantries do not.
#
# (3) TOOL_CALL combines the structured ``tool_calls`` field with tool-call
# text in ``content`` (XML / JSON / Python call list), since some handlers
# (hermes XML, llama3_json) deliver calls only as text.
#
# CONFIRM uses a τ²-specific phrase set: customer-service confirmations differ
# from BFCL's filesystem ones.

_TAU2_GREETING_QUESTION_PATTERNS: tuple[str, ...] = (
    # Opening greetings
    "how can i help",
    "how may i help",
    "how can i assist",
    "how may i assist",
    "what can i help",
    "what can i do for you",
    "what brings you",
    # Closing / between-task acknowledgments
    "is there anything else",
    "anything else i can",
    "anything else you need",
    "anything else for you",
)

# τ²-customer-service-specific extensions to the BFCL ASK_PHRASES list.
# τ² scenarios commonly use imperative info-request verbs ("please send",
# "send me", "give me") that filesystem-domain BFCL did not enumerate.
# Treated as substantive-ASK markers identically to the BFCL list.
_TAU2_ASK_PHRASES_EXTENSION: tuple[str, ...] = (
    "please send",
    "could you send",
    "kindly send",
    "send me",
    "give me",
    "i'd like to know",
    "i would like to know",
)

_TAU2_CONFIRM_PHRASES: tuple[str, ...] = (
    # passive past-tense action completion
    "has been cancelled", "has been canceled",
    "have been cancelled", "have been canceled",
    "has been booked", "have been booked",
    "has been updated", "have been updated",
    "has been submitted", "have been submitted",
    "has been issued", "have been issued",
    "has been processed", "have been processed",
    "has been modified", "have been modified",
    "has been confirmed", "have been confirmed",
    "has been refunded", "have been refunded",
    "has been exchanged", "have been exchanged",
    "has been returned", "have been returned",
    "has been added", "have been added",
    "has been changed", "have been changed",
    "has been completed", "have been completed",
    # first-person past-tense completions
    "i've cancelled", "i've canceled",
    "i've booked", "i've updated", "i've submitted",
    "i've issued", "i've processed", "i've modified",
    "i've confirmed", "i've refunded", "i've exchanged",
    "i've returned", "i've added", "i've changed",
    "i have cancelled", "i have canceled",
    "i have booked", "i have updated", "i have submitted",
    "i have issued", "i have processed", "i have modified",
    "i have confirmed", "i have refunded", "i have exchanged",
    "i have returned",
    # tau2 customer-service idioms
    "request is in", "request has been",
    "transferred to a human agent",
    "transferring you to a human", "transferring to a human",
    "your refund will be", "your refund has been",
)


def _matches_tau2_confirm(normalized: str | None) -> bool:
    """Match τ²-customer-service CONFIRM phrases on a quote-normalized
    string (typographic apostrophes already translated to ASCII)."""
    if not normalized:
        return False
    lower = normalized.lower()
    return any(phrase in lower for phrase in _TAU2_CONFIRM_PHRASES)


def _has_substantive_ask_tau2(normalized: str | None) -> bool:
    """True iff *normalized* contains an ASK signal beyond a greeting/closing.

    Expects *normalized* to have been pre-processed by
    :func:`_normalize_typographic_quotes` so curly apostrophes /
    quotation marks are translated to ASCII (RLHF-tuned models
    routinely emit U+2019, which otherwise breaks contractions like
    ``can't`` / ``what's``).

    Two paths to True:

    (a) Any ASK_PHRASE imperative (``please tell``, ``could you``,
    ``please provide``, ...) appears in the text. These imperatives are
    substantive by construction --- they explicitly request information
    or action from the user.

    (b) A "?"-terminated sentence appears whose lowercase form does not
    match any opening-greeting or closing-acknowledgment pattern in
    :data:`_TAU2_GREETING_QUESTION_PATTERNS`. This catches frontier-style
    open-ended clarifying questions ("What would you like to do?")
    while excluding "How can I help you today?" and "Is there anything
    else?".
    """
    if not normalized:
        return False
    lower = normalized.lower()
    if any(phrase in lower for phrase in ASK_PHRASES):
        return True
    if any(phrase in lower for phrase in _TAU2_ASK_PHRASES_EXTENSION):
        return True
    if "?" not in normalized:
        return False
    for sentence in re.split(r"[.!]+\s*", normalized):
        if "?" not in sentence:
            continue
        s_lower = sentence.lower()
        if any(g in s_lower for g in _TAU2_GREETING_QUESTION_PATTERNS):
            continue
        return True
    return False


def _detect_actions_tau2(msg: dict[str, Any]) -> set[Action]:
    """Multi-label τ²-native action detection for a single assistant message.

    Independent checks for each class (no priority cascade). A message
    may carry zero, one, or multiple labels.

    - ``TOOL_CALL``: structured ``tool_calls`` field non-empty, OR
      canonical text-format tool call in ``content`` (handles xLAM
      llama3_json, Hammer hermes XML, etc.).
    - ``REFUSE``: any REFUSE_PHRASES match (``cannot``, ``can't``,
      ``unable to``, ...).
    - ``ASK``: substantive ask via :func:`_has_substantive_ask_tau2`
      (imperative ASK_PHRASES OR non-greeting "?" question).
    - ``CONFIRM``: τ²-specific completion phrase set.

    A single message can carry e.g. REFUSE + ASK ("I can't do X,
    please send Y"), or TOOL_CALL + CONFIRM (call followed by
    completion announcement).
    """
    actions: set[Action] = set()

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        actions.add(Action.TOOL_CALL)

    content = _unfence_markdown(msg.get("content"))
    if not content:
        return actions

    # Tool-spec handlers (Hammer hermes XML, xLAM llama3_json, etc.)
    # deliver tool calls as canonical text in ``content`` rather than
    # populating the ``tool_calls`` field; catch those structurally.
    if Action.TOOL_CALL not in actions and is_tool_call_emission(content):
        actions.add(Action.TOOL_CALL)

    # Normalize typographic quotes (U+2019 etc.) to ASCII before
    # phrase matching. RLHF-tuned and frontier models routinely emit
    # ``can't`` / ``don't`` / ``what's`` with curly apostrophes; without
    # normalization, contractions in REFUSE_PHRASES / ASK_PHRASES fail
    # to match.
    normalized = _normalize_typographic_quotes(content)
    lower = normalized.lower()

    has_substantive_ask = _has_substantive_ask_tau2(normalized)
    has_refuse_language = any(phrase in lower for phrase in REFUSE_PHRASES)

    if has_substantive_ask:
        actions.add(Action.ASK)

    # REFUSE fires only when refuse language is NOT accompanied by a
    # substantive information-seeking ask. Messages like "I can't
    # proceed without your ID, please tell me X" are functionally an
    # ASK with a conditional explainer --- the model is continuing the
    # conversation by requesting info, not declining the task. True
    # REFUSE is reserved for absolute decline without an info-request
    # ("I cannot help with that. Please contact a human agent.";
    # "I don't have a tool for that.").
    if has_refuse_language and not has_substantive_ask:
        actions.add(Action.REFUSE)

    if _matches_tau2_confirm(normalized):
        actions.add(Action.CONFIRM)

    return actions


def _summarize_simulation(
    sim: dict[str, Any], write_tool_set: set[str]
) -> dict[str, Any]:
    emit_set: set[Action] = set()
    any_write_call = False
    for msg in sim.get("messages") or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        emit_set |= _detect_actions_tau2(msg)
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("name") in write_tool_set:
                any_write_call = True
                break
    reward_info = sim.get("reward_info") or {}
    return {
        "task_id": str(sim.get("task_id")),
        "emit_set": emit_set,
        "any_write_call": any_write_call,
        "pass_hit": pass_at_one_hit(reward_info),
        "action_gar_hit": failed_action_count(reward_info) == 0,
    }


def _aggregate(
    sim_rows: list[dict[str, Any]],
    task_labels: dict[str, list[str]],
) -> dict[str, Any]:
    scored = sim_rows
    n_scored = len(scored)
    if n_scored == 0:
        return {}

    # Subset masks
    refuse_tasks = [r for r in scored if "REFUSE" in task_labels.get(r["task_id"], [])]
    ask_tasks = [r for r in scored if "ASK" in task_labels.get(r["task_id"], [])]
    tool_tasks = [r for r in scored if "TOOL_CALL" in task_labels.get(r["task_id"], [])]

    def _rate(num: int, denom: int) -> float | None:
        return None if denom == 0 else num / denom

    capability: dict[str, dict[str, Any]] = {}
    for cls in CAPABILITY_CLASSES:
        n_emit = sum(1 for r in scored if cls in r["emit_set"])
        capability[cls.value] = {
            "tasks_with_emission": n_emit,
            "tasks_total": n_scored,
            "rate": _rate(n_emit, n_scored),
        }

    calibration: dict[str, dict[str, Any]] = {
        # REFUSE uses τ²-bench's structural signal: on a REFUSE-required
        # task (no gold actions, so the database must stay unchanged) the
        # agent refused iff it made no WRITE tool call. Verbal refusal is not
        # what τ²-bench scores, so text-detected REFUSE counts only toward
        # capability above. ASK comes from the text-detected emit set (τ²
        # has no structural ASK signal); TOOL_CALL from the reference
        # ``action_checks`` (action_gar_hit).
        "refuse": {
            "denom": len(refuse_tasks),
            "gar_hits": sum(1 for r in refuse_tasks if not r["any_write_call"]),
            "acc_hits": sum(1 for r in refuse_tasks if r["pass_hit"]),
        },
        "ask": {
            "denom": len(ask_tasks),
            "gar_hits": sum(1 for r in ask_tasks if Action.ASK in r["emit_set"]),
            "acc_hits": sum(1 for r in ask_tasks if r["pass_hit"]),
        },
        "tool_call": {
            "denom": len(tool_tasks),
            "gar_hits": sum(1 for r in tool_tasks if r["action_gar_hit"]),
            "acc_hits": sum(1 for r in tool_tasks if r["pass_hit"]),
        },
    }
    for entry in calibration.values():
        entry["gar"] = _rate(entry["gar_hits"], entry["denom"])
        entry["acc"] = _rate(entry["acc_hits"], entry["denom"])
        if entry["gar"] is not None and entry["acc"] is not None:
            entry["delta"] = entry["gar"] - entry["acc"]
        else:
            entry["delta"] = None

    pass_hits = sum(1 for r in scored if r["pass_hit"])
    return {
        "scored_simulations": n_scored,
        "pass_at_1": _rate(pass_hits, n_scored),
        "pass_hits": pass_hits,
        "capability": capability,
        "calibration": calibration,
    }


def summarize_run(
    run: RunInfo,
    task_labels: dict[str, list[str]],
    write_tool_set: set[str],
) -> dict[str, Any]:
    data = json.loads(run.path.read_text())
    sims = data.get("simulations") or []
    sim_rows: list[dict[str, Any]] = []
    n_unscored = 0
    for sim in sims:
        if not isinstance(sim, dict):
            continue
        if not is_scored_simulation(sim):
            n_unscored += 1
            continue
        sim_rows.append(_summarize_simulation(sim, write_tool_set))
    agg = _aggregate(sim_rows, task_labels)
    return {
        "domain": run.domain,
        "family": run.family,
        "method": run.method,
        "path": str(run.path),
        "unscored_simulations": n_unscored,
        **agg,
    }


def discover_runs(roots: Iterable[Path]) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for root in roots:
        for path in sorted(root.glob("*/results.json")):
            info = parse_run_dir(path)
            if info is not None:
                runs.append(info)
    return sorted(runs, key=lambda r: (r.domain, r.family, r.method))


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}"


_CSV_FIELDS = (
    "domain",
    "family",
    "method",
    "scored_simulations",
    "pass_at_1_pct",
    "cap_tool_call_pct",
    "cap_ask_pct",
    "cap_refuse_pct",
    "cap_confirm_pct",
    "calib_tool_call_gar_pct",
    "calib_tool_call_acc_pct",
    "calib_tool_call_delta_pp",
    "calib_tool_call_denom",
    "calib_ask_gar_pct",
    "calib_ask_acc_pct",
    "calib_ask_delta_pp",
    "calib_ask_denom",
    "calib_refuse_gar_pct",
    "calib_refuse_acc_pct",
    "calib_refuse_delta_pp",
    "calib_refuse_denom",
    "path",
)


def _delta_pp(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:+.1f}"


def _csv_row(row: dict[str, Any]) -> dict[str, str]:
    cap = row.get("capability") or {}
    cal = row.get("calibration") or {}
    def _cls(name: str) -> dict[str, Any]:
        return cal.get(name) or {}
    return {
        "domain": row["domain"],
        "family": row["family"],
        "method": row["method"],
        "scored_simulations": str(row.get("scored_simulations") or 0),
        "pass_at_1_pct": _pct(row.get("pass_at_1")),
        "cap_tool_call_pct": _pct(cap.get("tool_call", {}).get("rate")),
        "cap_ask_pct": _pct(cap.get("ask", {}).get("rate")),
        "cap_refuse_pct": _pct(cap.get("refuse", {}).get("rate")),
        "cap_confirm_pct": _pct(cap.get("confirm", {}).get("rate")),
        "calib_tool_call_gar_pct": _pct(_cls("tool_call").get("gar")),
        "calib_tool_call_acc_pct": _pct(_cls("tool_call").get("acc")),
        "calib_tool_call_delta_pp": _delta_pp(_cls("tool_call").get("delta")),
        "calib_tool_call_denom": str(_cls("tool_call").get("denom") or 0),
        "calib_ask_gar_pct": _pct(_cls("ask").get("gar")),
        "calib_ask_acc_pct": _pct(_cls("ask").get("acc")),
        "calib_ask_delta_pp": _delta_pp(_cls("ask").get("delta")),
        "calib_ask_denom": str(_cls("ask").get("denom") or 0),
        "calib_refuse_gar_pct": _pct(_cls("refuse").get("gar")),
        "calib_refuse_acc_pct": _pct(_cls("refuse").get("acc")),
        "calib_refuse_delta_pp": _delta_pp(_cls("refuse").get("delta")),
        "calib_refuse_denom": str(_cls("refuse").get("denom") or 0),
        "path": row["path"],
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "domain",
        "family",
        "method",
        "n",
        "pass@1",
        "cap(TC/ASK/REF/CONF)",
        "calib(TC/ASK/REF)",
    )
    print("\t".join(headers))
    for row in rows:
        cap = row.get("capability") or {}
        cal = row.get("calibration") or {}
        cap_str = (
            f"{_pct(cap.get('tool_call', {}).get('rate'))}/"
            f"{_pct(cap.get('ask', {}).get('rate'))}/"
            f"{_pct(cap.get('refuse', {}).get('rate'))}/"
            f"{_pct(cap.get('confirm', {}).get('rate'))}"
        )
        cal_str = (
            f"{_pct(cal.get('tool_call', {}).get('gar'))}/"
            f"{_pct(cal.get('ask', {}).get('gar'))}/"
            f"{_pct(cal.get('refuse', {}).get('gar'))}"
        )
        print(
            "\t".join(
                [
                    row["domain"],
                    row["family"],
                    row["method"],
                    str(row.get("scored_simulations") or 0),
                    _pct(row.get("pass_at_1")),
                    cap_str,
                    cal_str,
                ]
            )
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True, nargs="+", type=Path)
    ap.add_argument(
        "--task-classification",
        required=True,
        type=Path,
        help="Output of scripts/classify_tau2_tasks.py",
    )
    ap.add_argument(
        "--tau2-root",
        required=True,
        type=Path,
        help="sierra-research/tau2-bench checkout for WRITE-tool detection.",
    )
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--output-csv", type=Path)
    args = ap.parse_args()

    task_classes_by_domain: dict[str, dict[str, list[str]]] = json.loads(
        args.task_classification.read_text()
    )
    write_tool_set: dict[str, set[str]] = {
        dom: write_tools(args.tau2_root.resolve(), dom) for dom in DOMAINS
    }

    rows: list[dict[str, Any]] = []
    for run in discover_runs(args.runs):
        labels = task_classes_by_domain.get(run.domain) or {}
        rows.append(summarize_run(run, labels, write_tool_set[run.domain]))

    if not rows:
        roots = ", ".join(str(p) for p in args.runs)
        raise SystemExit(f"No tau2 results.json files found under: {roots}")

    _print_table(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n"
        )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(_csv_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
