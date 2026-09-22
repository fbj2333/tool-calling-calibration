"""BFCL multi-turn driver with the SRI and CRI probes.

Runs upstream BFCL (the ``bfcl-eval`` package) on local checkpoints in native
function-calling mode (``python -m bfcl_calibration.evaluation.bfcl``). The
probes are installed into BFCL's multi-turn loop by ``_apply_sri_patches`` and
selected with environment variables: SRI (``_sri``, the halt seam) with
``BFCL_SRI_MODE`` / ``BFCL_SRI_VARIANT``, CRI (``_cri``, the call seam) with
``BFCL_CRI_MODE`` / ``BFCL_CRI_VARIANT``. ``scripts/run_bfcl_api.py`` installs
the same loop for API models.

Module map: ``_categories`` (categories, run subsets), ``_score`` (upstream
loading, model registration, summary), ``_runner`` (entry point),
``_patches`` (vLLM server arguments), ``_gptoss`` + ``_gptoss_handlers``
(gpt-oss handler), ``_state`` (tool-call parsing and state tracking for the
probes), ``_sri`` + ``_sri_loop`` and ``_cri`` (the probes).
"""
from __future__ import annotations

from ._categories import MULTI_TURN_CATEGORIES, _build_subset_mapping  # noqa: F401
from ._cri import (  # noqa: F401
    _CRI_HEADER,
    _cri_active,
    _cri_detect,
    _cri_enabled,
    _cri_evaluate,
    _cri_mode,
    _cri_retry_instruction,
    _cri_variant,
)
from ._patches import _patch_vllm_server_args  # noqa: F401
from ._score import (  # noqa: F401
    BFCLOfficialHandles,
    CategoryScoreSummary,
    _ensure_bfcl_vendor_root,
    _load_bfcl_handles,
    _preflight_backend_requirements,
    _register_runtime_model,
    _runtime_model_name,
)
from ._sri import SRIRuntime, _sri_mode, _sri_variant, is_asking_response  # noqa: F401
from ._sri_loop import _apply_sri_patches, _handle_overaction_candidate  # noqa: F401
from ._runner import _build_parser, main, run_bfcl_benchmark  # noqa: F401


__all__ = [
    "MULTI_TURN_CATEGORIES",
    "is_asking_response",
    "main",
    "run_bfcl_benchmark",
]

