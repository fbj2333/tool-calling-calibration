#!/usr/bin/env python3
"""Run BFCL multi-turn on an OpenAI-compatible model, optionally under SRI or CRI.

Registers ``--model-id`` with upstream BFCL's OpenAI-compatible function-calling
handler, then runs upstream generation and scoring into ``--out-dir``
(``bfcl_results/`` and ``bfcl_scores/``). The endpoint and key come from
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``. Setting ``BFCL_SRI_MODE=sri_only`` or
``BFCL_CRI_MODE=cri_only`` installs the corresponding probe in BFCL's
multi-turn loop; with neither set the run is plain upstream BFCL.

Examples::

    export OPENAI_API_KEY=... OPENAI_BASE_URL=https://<endpoint>/v1
    python scripts/run_bfcl_api.py --model-id gpt-5.4-2026-03-05 \
        --reasoning-effort high --out-dir runs/gpt-5.4
    BFCL_SRI_MODE=sri_only python scripts/run_bfcl_api.py \
        --model-id gpt-5.4-2026-03-05 --reasoning-effort high --out-dir runs/gpt-5.4-sri
    python -m bfcl_calibration.analysis.gar \
        --results runs/gpt-5.4/bfcl_results --scores runs/gpt-5.4/bfcl_scores
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CATEGORIES = [
    "multi_turn_base",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]


def install_reasoning_effort(handler_cls: Any, level: str) -> None:
    """Send ``reasoning_effort`` on every chat.completions call of the handler.

    The upstream OpenAI handler does not forward it. Reasoning models accept
    only the default temperature, so the call's temperature is set to 1.
    """
    original_query = handler_cls._query_FC

    def _query_fc(self, inference_data):  # noqa: ANN001
        client = self.client
        if not getattr(client, "_reasoning_effort_set", False):
            create = client.chat.completions.create

            def _create(**kwargs):
                kwargs.setdefault("reasoning_effort", level)
                kwargs["temperature"] = 1
                return create(**kwargs)

            client.chat.completions.create = _create
            client._reasoning_effort_set = True
        return original_query(self, inference_data)

    handler_cls._query_FC = _query_fc


def write_first_n_ids(load_dataset_entry: Any, project_root: Path, categories: list[str], n: int) -> None:
    """Restrict upstream generation to the first ``n`` cases of each category."""
    ids = {cat: [entry["id"] for entry in load_dataset_entry(cat)][:n] for cat in categories}
    (project_root / "test_case_ids_to_generate.json").write_text(json.dumps(ids, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-id", required=True, help="model id exactly as the endpoint serves it")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"],
                    help="send reasoning_effort (gpt-5 family); omit for the server default")
    ap.add_argument("--num-threads", type=int, default=1)
    ap.add_argument("--limit", type=int, help="run only the first N cases of each category")
    ap.add_argument("--no-eval", action="store_true", help="generate only; skip BFCL scoring")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Upstream resolves result/score directories against BFCL_PROJECT_ROOT at import time.
    os.environ["BFCL_PROJECT_ROOT"] = str(out_dir)

    generation = importlib.import_module("bfcl_eval._llm_response_generation")
    evaluation = importlib.import_module("bfcl_eval.eval_checker.eval_runner")
    model_config = importlib.import_module("bfcl_eval.constants.model_config")
    base_handler = importlib.import_module("bfcl_eval.model_handler.base_handler")
    utils = importlib.import_module("bfcl_eval.utils")
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    from bfcl_calibration.evaluation.bfcl import _apply_sri_patches, _cri_enabled, _cri_mode, _sri_mode

    name = f"{args.model_id}-FC"
    config = model_config.ModelConfig(
        model_name=args.model_id,
        display_name=args.model_id,
        url=args.model_id,
        org="",
        license="proprietary",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=False,
    )
    model_config.MODEL_CONFIG_MAPPING[name] = config
    model_config.MODEL_CONFIG_MAPPING[name.replace("_", "/")] = config
    if args.reasoning_effort:
        install_reasoning_effort(OpenAICompletionsHandler, args.reasoning_effort)
    _apply_sri_patches(base_handler)
    if (_sri_mode() != "off" or _cri_enabled()) and not os.environ.get("BFCL_SRI_LOG_PATH", "").strip():
        os.environ["BFCL_SRI_LOG_PATH"] = str(out_dir / "sri_event_log.jsonl")
    print(f"model {name}: SRI mode={_sri_mode()}, CRI mode={_cri_mode()}")

    run_ids = args.limit is not None
    if run_ids:
        write_first_n_ids(utils.load_dataset_entry, out_dir, args.categories, args.limit)

    generation.main(SimpleNamespace(
        model=[name],
        test_category=list(args.categories),
        temperature=args.temperature,
        include_input_log=False,
        exclude_state_log=False,
        num_threads=args.num_threads,
        num_gpus=1,
        backend="vllm",  # unused by API handlers
        gpu_memory_utilization=0.9,
        result_dir="bfcl_results",
        run_ids=run_ids,
        allow_overwrite=False,  # resume: cases already in the result file are skipped
        skip_server_setup=True,
        local_model_path=None,
        lora_modules=None,
        enable_lora=False,
        max_lora_rank=None,
    ))
    if not args.no_eval:
        evaluation.main([name], list(args.categories), "bfcl_results", "bfcl_scores", partial_eval=run_ids)
    print(f"done: {out_dir}/bfcl_results, {out_dir}/bfcl_scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
