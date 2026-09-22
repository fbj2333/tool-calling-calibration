"""Run ``tau2 run`` with a configurable natural-language-assertion judge.

tau2-bench fixes the model that grades natural-language assertions (used by
retail tasks) in ``tau2.config``, and ``tau2.evaluator.evaluator_nl_assertions``
imports it by value. ``TAU2_NL_ASSERTIONS_LLM`` (a litellm model string) and
``TAU2_NL_ASSERTIONS_ARGS_JSON`` (a JSON dict of call arguments) replace it in
both modules; every command-line argument is passed on to ``tau2 run``::

    python -m bfcl_calibration.evaluation.tau2_run --domain retail --agent-llm ... --user-llm ...
"""
from __future__ import annotations

import json
import os
import sys


def apply_nl_judge_override() -> None:
    model = os.environ.get("TAU2_NL_ASSERTIONS_LLM", "").strip()
    if not model:
        return
    args = json.loads(os.environ.get("TAU2_NL_ASSERTIONS_ARGS_JSON", "").strip() or "{}")

    import tau2.config as tau2_config
    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    for module in (tau2_config, nl_eval):
        module.DEFAULT_LLM_NL_ASSERTIONS = model
        module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args


def main() -> int:
    apply_nl_judge_override()
    sys.argv = ["tau2", "run", *sys.argv[1:]]
    from tau2.cli import main as tau2_main

    return tau2_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
