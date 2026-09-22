from __future__ import annotations

from types import SimpleNamespace

import pytest

from bfcl_calibration.evaluation import bfcl

UPSTREAM_COMMAND = ["vllm", "serve", "MODEL"]  # the form bfcl-eval's OSSHandler launches


def _serve_command(monkeypatch: pytest.MonkeyPatch, base: list[str], **env: str) -> list[str]:
    """Run a patched fake ``spin_up_local_server`` and return the command it launched."""
    for name in ("BFCL_VLLM_EXTRA_ARGS", "BFCL_SEED"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    commands: list[list[str]] = []
    fake_module = SimpleNamespace()

    class FakeOSSHandler:
        def spin_up_local_server(
            self,
            num_gpus,
            gpu_memory_utilization,
            backend,
            skip_server_setup,
            local_model_path,
        ):
            fake_module.subprocess.Popen(list(base))

    fake_module.OSSHandler = FakeOSSHandler
    fake_module.subprocess = SimpleNamespace(Popen=lambda command, *a, **k: commands.append(command))

    bfcl._patch_vllm_server_args(fake_module)
    FakeOSSHandler().spin_up_local_server(1, 0.7, "vllm", False, "MODEL")
    assert len(commands) == 1
    return commands[0]


def test_command_is_unchanged_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _serve_command(monkeypatch, UPSTREAM_COMMAND) == UPSTREAM_COMMAND


def test_extra_args_and_seed_are_appended(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _serve_command(
        monkeypatch, UPSTREAM_COMMAND, BFCL_VLLM_EXTRA_ARGS="--max-model-len 8192 --enforce-eager", BFCL_SEED="43"
    )
    assert command == UPSTREAM_COMMAND + ["--max-model-len", "8192", "--enforce-eager", "--seed", "43"]


def test_explicit_seed_in_extra_args_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    command = _serve_command(monkeypatch, UPSTREAM_COMMAND, BFCL_VLLM_EXTRA_ARGS="--seed 44", BFCL_SEED="42")
    assert command == UPSTREAM_COMMAND + ["--seed", "44"]


def test_module_form_is_also_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    base = ["python", "-m", "vllm.entrypoints.cli.main", "serve", "MODEL"]
    assert _serve_command(monkeypatch, base, BFCL_SEED="42") == base + ["--seed", "42"]
