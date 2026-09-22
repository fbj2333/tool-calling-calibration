"""Run tau2-bench with the SRI-lite agent.

Importing ``bfcl_calibration.evaluation.tau2_sri_lite`` registers
``SRILiteLLMAgent`` as the factory behind tau2's default ``llm_agent``; the
command-line arguments then go to ``tau2 run`` exactly as in
``bfcl_calibration.evaluation.tau2_run`` (including its NL-assertion judge
override). ``SRI_LITE_VARIANT`` selects the variant: ``sri_lite`` (default),
``ablation_no_state``, or ``baseline`` (probe off)::

    SRI_LITE_VARIANT=sri_lite python -m bfcl_calibration.evaluation.tau2_sri_lite_launcher \\
        --domain retail --agent-llm <model> --user-llm <model> ...
"""
from __future__ import annotations

from bfcl_calibration.evaluation import tau2_run

tau2_run.apply_nl_judge_override()

import bfcl_calibration.evaluation.tau2_sri_lite  # noqa: E402,F401  (registers the agent)


if __name__ == "__main__":
    raise SystemExit(tau2_run.main())
