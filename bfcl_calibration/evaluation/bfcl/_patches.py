"""Extra arguments for the ``vllm serve`` command upstream BFCL launches.

``_patch_vllm_server_args`` appends ``BFCL_VLLM_EXTRA_ARGS`` and, when
``BFCL_SEED`` is set and those arguments carry no ``--seed``, ``--seed
$BFCL_SEED``. With neither variable set the command is upstream's.
"""
from __future__ import annotations

import os
import shlex
from typing import Any


def _vllm_serve_extra_args() -> list[str]:
    extra_args = shlex.split(os.environ.get("BFCL_VLLM_EXTRA_ARGS", ""))
    seed = os.environ.get("BFCL_SEED", "").strip()
    if seed and not any(arg == "--seed" or arg.startswith("--seed=") for arg in extra_args):
        extra_args += ["--seed", seed]
    return extra_args


def _is_vllm_serve(command: Any) -> bool:
    """``vllm serve ...`` (upstream's form) or ``python -m vllm.entrypoints.cli.main serve ...``."""
    if not isinstance(command, list) or "serve" not in command:
        return False
    return command[:2] == ["vllm", "serve"] or "vllm.entrypoints.cli.main" in command


def _patch_vllm_server_args(base_oss_module: Any) -> None:
    cls = base_oss_module.OSSHandler
    if getattr(cls.spin_up_local_server, "_bfclcalib_vllm_args_patched", False):
        return

    original_spin_up = cls.spin_up_local_server

    def _patched_spin_up(
        self,
        num_gpus: int,
        gpu_memory_utilization: float,
        backend: str,
        skip_server_setup: bool,
        local_model_path: str | None,
        *args,
        **kwargs,
    ):
        extra_args = _vllm_serve_extra_args()
        original_popen = base_oss_module.subprocess.Popen

        def _patched_popen(command, *popen_args, **popen_kwargs):
            if extra_args and _is_vllm_serve(command):
                command = command + extra_args
            return original_popen(command, *popen_args, **popen_kwargs)

        base_oss_module.subprocess.Popen = _patched_popen
        try:
            return original_spin_up(
                self,
                num_gpus,
                gpu_memory_utilization,
                backend,
                skip_server_setup,
                local_model_path,
                *args,
                **kwargs,
            )
        finally:
            base_oss_module.subprocess.Popen = original_popen

    _patched_spin_up._bfclcalib_vllm_args_patched = True  # type: ignore[attr-defined]
    cls.spin_up_local_server = _patched_spin_up
