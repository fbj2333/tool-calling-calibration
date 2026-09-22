"""``run_bfcl_benchmark`` and the ``python -m bfcl_calibration.evaluation.bfcl`` CLI.

Registers a local checkpoint with upstream BFCL, runs upstream generation and
scoring on the multi-turn categories, and writes ``summary.{json,md}``. BFCL's
``bfcl_results/`` and ``bfcl_scores/`` live in the output directory (symlinked
from a temporary BFCL project root), so results are written there case by case
and an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from ._categories import (
    _BFCL_RUN_IDS_FILENAME,
    _build_subset_mapping,
    _load_run_ids_mapping,
    _resolve_categories,
    _write_subset_mapping,
)
from ._cri import _cri_enabled, _cri_mode, _cri_variant
from ._score import (
    _BFCL_RESULTS_DIRNAME,
    _BFCL_SCORES_DIRNAME,
    BFCLOfficialHandles,
    _register_runtime_model,
    _write_summary_files,
)
from ._sri import _sri_mode, _sri_variant


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BFCL multi-turn on a local checkpoint in native function-calling "
            "mode, served with vLLM. BFCL_SRI_MODE=sri_only or BFCL_CRI_MODE=cri_only "
            "installs the corresponding probe."
        ),
    )
    parser.add_argument("--model-path", required=True, help="Local HF model directory to evaluate.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for bfcl_results/, bfcl_scores/, and summary.{json,md}.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=(
            "Multi-turn categories to evaluate, space- or comma-separated "
            "(default: all four multi_turn_* categories)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-category sample cap. Implemented via BFCL run-ids partial evaluation.",
    )
    parser.add_argument(
        "--run-ids-file",
        default=None,
        help="Optional JSON mapping of category -> list[test_id] for partial evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. Defaults to 0.0.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Tensor parallel GPU count.")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Optional BFCL generation thread count. If omitted, BFCL uses its backend default.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Regenerate cases already present in the output dir instead of skipping them.",
    )
    return parser


def run_bfcl_benchmark(
    *,
    model_path: str,
    output_dir: str,
    categories: Sequence[str] | None = None,
    max_samples: int | None = None,
    run_ids_file: str | None = None,
    temperature: float = 0.0,
    gpu_memory_utilization: float = 0.9,
    num_gpus: int = 1,
    num_threads: int | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    if max_samples is not None and run_ids_file is not None:
        raise ValueError("--max-samples and --run-ids-file are mutually exclusive.")
    if num_gpus < 1:
        raise ValueError("--num-gpus must be >= 1.")

    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {resolved_model_path}")

    run_ids_mapping = None
    if run_ids_file is not None:
        resolved_run_ids_file = Path(run_ids_file).expanduser().resolve()
        if not resolved_run_ids_file.exists():
            raise FileNotFoundError(f"Run-ids file not found: {resolved_run_ids_file}")
        run_ids_mapping = _load_run_ids_mapping(resolved_run_ids_file)

    normalized_categories = _resolve_categories(categories, run_ids_mapping)

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    if (_sri_mode() != "off" or _cri_enabled()) and not os.environ.get("BFCL_SRI_LOG_PATH", "").strip():
        os.environ["BFCL_SRI_LOG_PATH"] = str(resolved_output_dir / "sri_event_log.jsonl")

    tmp_project_root = Path(tempfile.mkdtemp(prefix="bfclcalib_bfcl_run_"))
    resolved_results_dir = resolved_output_dir / _BFCL_RESULTS_DIRNAME
    resolved_scores_dir = resolved_output_dir / _BFCL_SCORES_DIRNAME
    resolved_results_dir.mkdir(parents=True, exist_ok=True)
    resolved_scores_dir.mkdir(parents=True, exist_ok=True)
    (tmp_project_root / _BFCL_RESULTS_DIRNAME).symlink_to(
        resolved_results_dir, target_is_directory=True
    )
    (tmp_project_root / _BFCL_SCORES_DIRNAME).symlink_to(
        resolved_scores_dir, target_is_directory=True
    )
    # Declare outside try so post-finally code can access these even though
    # they're assigned inside the try block.
    handles: BFCLOfficialHandles
    runtime_model_name: str
    partial_eval = False
    # Dynamic lookup via parent package so test monkeypatches at
    # ``bfcl._load_bfcl_handles`` / ``bfcl._preflight_backend_requirements``
    # reach the actual call sites; a static ``from ._score import ...``
    # would bind originals at import time and miss the patch.
    import sys as _sys
    from . import __name__ as _pkg_name

    _pkg = _sys.modules[_pkg_name]
    try:
        handles = _pkg._load_bfcl_handles(tmp_project_root)
        _pkg._preflight_backend_requirements()
        subset_mapping = _build_subset_mapping(
            handles,
            categories=normalized_categories,
            max_samples=max_samples,
            run_ids_mapping=run_ids_mapping,
        )
        partial_eval = subset_mapping is not None
        if subset_mapping is not None:
            _write_subset_mapping(tmp_project_root, subset_mapping)

        runtime_model_name = _register_runtime_model(handles, model_path=resolved_model_path)

        generation_args = SimpleNamespace(
            model=[runtime_model_name],
            test_category=list(normalized_categories),
            temperature=temperature,
            include_input_log=False,
            exclude_state_log=False,
            num_threads=num_threads,
            num_gpus=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            backend="vllm",
            skip_server_setup=False,
            local_model_path=str(resolved_model_path),
            result_dir=_BFCL_RESULTS_DIRNAME,
            allow_overwrite=allow_overwrite,
            run_ids=subset_mapping is not None,
            enable_lora=False,
            max_lora_rank=None,
            lora_modules=None,
        )
        handles.generation_main(generation_args)

        handles.evaluation_main(
            [runtime_model_name],
            list(normalized_categories),
            _BFCL_RESULTS_DIRNAME,
            _BFCL_SCORES_DIRNAME,
            partial_eval=subset_mapping is not None,
        )

        if subset_mapping is not None and (tmp_project_root / _BFCL_RUN_IDS_FILENAME).exists():
            shutil.copy(
                tmp_project_root / _BFCL_RUN_IDS_FILENAME,
                resolved_output_dir / _BFCL_RUN_IDS_FILENAME,
            )

    finally:
        shutil.rmtree(tmp_project_root, ignore_errors=True)

    return _write_summary_files(
        output_dir=resolved_output_dir,
        model_path=resolved_model_path,
        runtime_model_name=runtime_model_name,
        categories=normalized_categories,
        partial_eval=partial_eval,
        handles=handles,
        run_config={
            "temperature": temperature,
            "gpu_memory_utilization": gpu_memory_utilization,
            "num_gpus": num_gpus,
            "num_threads": num_threads,
            "bfcl_seed": os.environ.get("BFCL_SEED", ""),
            "bfcl_vllm_extra_args": os.environ.get("BFCL_VLLM_EXTRA_ARGS", ""),
            "bfcl_sri_mode": _sri_mode(),
            "bfcl_sri_variant": _sri_variant(),
            "bfcl_cri_mode": _cri_mode(),
            "bfcl_cri_variant": _cri_variant(),
            "bfcl_sri_log_path": os.environ.get("BFCL_SRI_LOG_PATH", ""),
            "sri_max_state_chars": os.environ.get("SRI_MAX_STATE_CHARS", "2400"),
            "sri_max_user_turns": os.environ.get("SRI_MAX_USER_TURNS", "4"),
            "sri_max_user_chars": os.environ.get("SRI_MAX_USER_CHARS", "700"),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_bfcl_benchmark(
        model_path=args.model_path,
        output_dir=args.output_dir,
        categories=args.categories,
        max_samples=args.max_samples,
        run_ids_file=args.run_ids_file,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpus=args.num_gpus,
        num_threads=args.num_threads,
        allow_overwrite=args.allow_overwrite,
    )
