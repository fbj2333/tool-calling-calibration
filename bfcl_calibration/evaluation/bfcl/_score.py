"""Upstream BFCL loading, local-model registration, and the run summary.

1. ``_load_bfcl_handles`` imports the installed ``bfcl-eval`` package, installs
   the vLLM server-argument patch and the SRI / CRI multi-turn loop, and returns
   the upstream entry points and handler classes as ``BFCLOfficialHandles``.
   ``_preflight_backend_requirements`` checks that vLLM is importable first.
2. ``_register_runtime_model`` adds a ``ModelConfig`` for a local checkpoint to
   upstream's ``MODEL_CONFIG_MAPPING``, choosing the handler upstream uses for
   that model: by name for Hammer, ToolACE, Watt-Tool, Granite-3, and xLAM, and
   otherwise by ``config.json`` ``model_type`` (Qwen3, gpt-oss, Llama-3.1).
3. ``_write_summary_files`` reads BFCL's per-category score files and writes
   ``summary.json`` and ``summary.md``.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._categories import _sanitize_slug


_BFCL_RESULTS_DIRNAME = "bfcl_results"
_BFCL_SCORES_DIRNAME = "bfcl_scores"
_BFCL_VERSION_PREFIX = "BFCL_v4"


@dataclass(frozen=True)
class BFCLOfficialHandles:
    generation_main: Any
    evaluation_main: Any
    ModelConfig: Any
    MODEL_CONFIG_MAPPING: dict[str, Any]
    QwenFCHandler: Any
    GPTOSSFCHandler: Any
    Llama31Handler: Any
    SalesforceLlamaHandler: Any
    SalesforceQwenHandler: Any
    LlamaHandler: Any
    HammerHandler: Any
    Granite3FCHandler: Any
    load_dataset_entry: Any
    get_directory_structure_by_category: Any


@dataclass(frozen=True)
class CategoryScoreSummary:
    category: str
    accuracy: float
    correct_count: int
    total_count: int
    failure_count: int
    error_type_breakdown: dict[str, int]
    score_file: str


def _bfcl_vendor_root() -> Path:
    """Directory holding the installed ``bfcl_eval`` package."""
    spec = importlib.util.find_spec("bfcl_eval")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError(f"bfcl-eval is not installed. {_benchmark_install_hint()}")
    return Path(next(iter(spec.submodule_search_locations))).parent


def _benchmark_install_hint() -> str:
    return 'Install the vLLM extra: pip install -e ".[vllm]"'


def _ensure_bfcl_vendor_root() -> Path:
    vendor_root = _bfcl_vendor_root()
    if not vendor_root.exists():
        raise FileNotFoundError(f"bfcl-eval package directory is missing: {vendor_root}")
    return vendor_root


def _purge_bfcl_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "bfcl_eval" or module_name.startswith("bfcl_eval."):
            sys.modules.pop(module_name, None)


def _load_bfcl_handles(project_root: Path) -> BFCLOfficialHandles:
    # Lazy imports keep _score importable without bfcl-eval installed.
    from ._gptoss_handlers import _make_gptoss_fc_handler
    from ._patches import _patch_vllm_server_args
    from ._sri_loop import _apply_sri_patches

    # Looked up through the parent package so that tests can monkeypatch
    # ``bfcl._ensure_bfcl_vendor_root``.
    from . import __name__ as _pkg_name
    import sys as _sys

    _pkg = _sys.modules[_pkg_name]
    vendor_root = _pkg._ensure_bfcl_vendor_root()
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)
    _purge_bfcl_modules()

    try:
        generation_module = importlib.import_module("bfcl_eval._llm_response_generation")
        eval_runner_module = importlib.import_module("bfcl_eval.eval_checker.eval_runner")
        model_config_module = importlib.import_module("bfcl_eval.constants.model_config")
        qwen_fc_module = importlib.import_module("bfcl_eval.model_handler.local_inference.qwen_fc")
        base_handler_module = importlib.import_module("bfcl_eval.model_handler.base_handler")
        llama31_module = importlib.import_module("bfcl_eval.model_handler.local_inference.llama_3_1")
        llama_module = importlib.import_module("bfcl_eval.model_handler.local_inference.llama")
        salesforce_llama_module = importlib.import_module("bfcl_eval.model_handler.local_inference.salesforce_llama")
        salesforce_qwen_module = importlib.import_module("bfcl_eval.model_handler.local_inference.salesforce_qwen")
        hammer_module = importlib.import_module("bfcl_eval.model_handler.local_inference.hammer")
        granite_3_module = importlib.import_module("bfcl_eval.model_handler.local_inference.granite_3")
        base_oss_module = importlib.import_module("bfcl_eval.model_handler.local_inference.base_oss_handler")
        utils_module = importlib.import_module("bfcl_eval.utils")
        _patch_vllm_server_args(base_oss_module)
        gptoss_fc_handler = _make_gptoss_fc_handler(base_oss_module.OSSHandler)
        _apply_sri_patches(base_handler_module)
    except Exception as exc:
        install_hint = _benchmark_install_hint()
        if isinstance(exc, ModuleNotFoundError):
            missing_name = getattr(exc, "name", None) or "unknown"
            raise RuntimeError(
                "Failed to import the official BFCL implementation from "
                f"{vendor_root} because Python dependency '{missing_name}' is missing. "
                f"{install_hint}"
            ) from exc
        raise RuntimeError(
            "Failed to import the official BFCL implementation from "
            f"{vendor_root}. {install_hint}"
        ) from exc

    return BFCLOfficialHandles(
        generation_main=generation_module.main,
        evaluation_main=eval_runner_module.main,
        ModelConfig=model_config_module.ModelConfig,
        MODEL_CONFIG_MAPPING=model_config_module.MODEL_CONFIG_MAPPING,
        QwenFCHandler=qwen_fc_module.QwenFCHandler,
        GPTOSSFCHandler=gptoss_fc_handler,
        Llama31Handler=llama31_module.LlamaHandler_3_1,
        SalesforceLlamaHandler=salesforce_llama_module.SalesforceLlamaHandler,
        SalesforceQwenHandler=salesforce_qwen_module.SalesforceQwenHandler,
        LlamaHandler=llama_module.LlamaHandler,
        HammerHandler=hammer_module.HammerHandler,
        Granite3FCHandler=granite_3_module.Granite3FCHandler,
        load_dataset_entry=utils_module.load_dataset_entry,
        get_directory_structure_by_category=utils_module.get_directory_structure_by_category,
    )


def _require_importable_module(module_name: str, *, install_hint: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", None) or module_name
        raise RuntimeError(
            f"BFCL backend preflight failed because Python dependency '{missing_name}' is missing. "
            f"{install_hint}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"BFCL backend preflight failed while importing '{module_name}'. {install_hint}"
        ) from exc


def _find_module_spec(module_name: str):
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return None


def _preflight_backend_requirements() -> None:
    install_hint = _benchmark_install_hint()
    _require_importable_module("transformers", install_hint=install_hint)
    if _find_module_spec("vllm.entrypoints.cli.main") is None:
        raise RuntimeError(
            "BFCL backend preflight failed because Python module 'vllm.entrypoints.cli.main' "
            f"is unavailable from the selected interpreter. {install_hint}"
        )


def _runtime_model_name(model_path: Path) -> str:
    resolved = model_path.resolve()
    path_hash = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return f"bfcl_calibration-local-{_sanitize_slug(model_path.name)}-{path_hash}-fc"


def _register_runtime_model(handles: BFCLOfficialHandles, *, model_path: Path) -> str:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config.json not found at {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type", "")).lower()
    model_identity = " ".join(
        str(config.get(key, "")) for key in ("_name_or_path", "architectures")
    ).lower()
    path_identity = str(model_path).lower()

    def _has(*keywords: str) -> bool:
        return any(k in model_identity or k in path_identity for k in keywords)

    # Name keywords take precedence over ``model_type``: Hammer (on Qwen2),
    # ToolACE and Watt-Tool (on Llama), Granite-3, and xLAM each have their own
    # upstream handler. The keywords follow upstream's MODEL_CONFIG_MAPPING ids.
    handler: Any
    is_fc_model = True
    if _has("hammer"):
        handler = handles.HammerHandler
    elif _has("toolace", "team-ace", "watt-tool", "watt-ai"):
        # Upstream serves ToolACE and Watt-Tool through the generic Llama
        # handler with is_fc_model=False.
        handler = handles.LlamaHandler
        is_fc_model = False
    elif _has("granite-3"):
        handler = handles.Granite3FCHandler
    elif _has("xlam", "salesforce"):
        if model_type.startswith("qwen"):
            handler = handles.SalesforceQwenHandler
        elif model_type.startswith("llama"):
            handler = handles.SalesforceLlamaHandler
        else:
            raise ValueError(
                "xLAM / Salesforce local model has unsupported config.json "
                f"model_type={model_type!r}; expected qwen* or llama* in {config_path}."
            )
    elif model_type == "gpt_oss":
        handler = handles.GPTOSSFCHandler
    elif model_type == "qwen3":
        handler = handles.QwenFCHandler
    elif model_type == "llama":
        handler = handles.Llama31Handler
    else:
        raise ValueError(
            "Local model not recognised. Supported via path/name keyword: "
            "hammer, toolace/team-ace, watt-tool/watt-ai, granite-3, "
            "xlam/salesforce. Supported via config.json model_type: qwen3, "
            f"gpt_oss, llama. Found model_type={model_type!r} in {config_path}."
        )

    runtime_model_name = _runtime_model_name(model_path)
    runtime_model_config = handles.ModelConfig(
        model_name=str(model_path),
        display_name=f"{model_path.name} (FC)",
        url=str(model_path),
        org="BFCL Calibration Local",
        license="unknown",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=is_fc_model,
        underscore_to_dot=False,
    )
    handles.MODEL_CONFIG_MAPPING[runtime_model_name] = runtime_model_config
    # Upstream BFCL stores result directories with "/" replaced by "_", then
    # maps every "_" back to "/" before looking up MODEL_CONFIG_MAPPING in the
    # scorer, so the runtime name ("bfcl_calibration-...") needs this alias.
    handles.MODEL_CONFIG_MAPPING[runtime_model_name.replace("_", "/")] = runtime_model_config
    return runtime_model_name


def _score_file_path(
    output_dir: Path,
    runtime_model_name: str,
    category: str,
    handles: BFCLOfficialHandles,
) -> Path:
    """Score file of one category, under upstream's per-category sub-directory."""
    subdir = handles.get_directory_structure_by_category(category)
    return (
        output_dir
        / _BFCL_SCORES_DIRNAME
        / runtime_model_name.replace("/", "_")
        / subdir
        / f"{_BFCL_VERSION_PREFIX}_{category}_score.json"
    )


def _load_score_summary(
    path: Path,
    *,
    category: str,
) -> CategoryScoreSummary:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Official BFCL score file is empty: {path}")

    header = lines[0]
    failures = lines[1:]
    error_type_breakdown: dict[str, int] = {}
    for failure in failures:
        error_type = failure.get("error_type") or failure.get("error", {}).get("error_type")
        error_key = str(error_type or "unknown")
        error_type_breakdown[error_key] = error_type_breakdown.get(error_key, 0) + 1

    return CategoryScoreSummary(
        category=category,
        accuracy=float(header["accuracy"]),
        correct_count=int(header["correct_count"]),
        total_count=int(header["total_count"]),
        failure_count=len(failures),
        error_type_breakdown=error_type_breakdown,
        score_file=str(path),
    )


def _write_summary_files(
    *,
    output_dir: Path,
    model_path: Path,
    runtime_model_name: str,
    categories: list[str],
    partial_eval: bool,
    handles: BFCLOfficialHandles,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    category_summaries = [
        _load_score_summary(
            _score_file_path(output_dir, runtime_model_name, category, handles),
            category=category,
        )
        for category in categories
    ]

    total_correct = sum(item.correct_count for item in category_summaries)
    total_count = sum(item.total_count for item in category_summaries)
    overall_accuracy = (total_correct / total_count) if total_count else 0.0

    summary = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_path": str(model_path),
        "runtime_model_name": runtime_model_name,
        "partial_eval": partial_eval,
        "results_dir": str(output_dir / _BFCL_RESULTS_DIRNAME),
        "scores_dir": str(output_dir / _BFCL_SCORES_DIRNAME),
        "run_config": dict(run_config or {}),
        "overall": {
            "accuracy": overall_accuracy,
            "correct_count": total_correct,
            "total_count": total_count,
        },
        "categories": [
            {
                "category": item.category,
                "accuracy": item.accuracy,
                "correct_count": item.correct_count,
                "total_count": item.total_count,
                "failure_count": item.failure_count,
                "error_type_breakdown": item.error_type_breakdown,
                "score_file": item.score_file,
            }
            for item in category_summaries
        ],
    }

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# BFCL multi-turn summary",
        "",
        f"- Model path: `{model_path}`",
        f"- Runtime model: `{runtime_model_name}`",
        f"- Partial eval: `{partial_eval}`",
        f"- Overall accuracy: `{overall_accuracy:.4f}` ({total_correct}/{total_count})",
        "",
        "| Category | Accuracy | Correct | Total | Failures |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in category_summaries:
        lines.append(
            f"| {item.category} | {item.accuracy:.4f} | {item.correct_count} | "
            f"{item.total_count} | {item.failure_count} |"
        )
        for error_type, count in sorted(item.error_type_breakdown.items()):
            lines.append(f"| {item.category}:{error_type} |  |  |  | {count} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary
