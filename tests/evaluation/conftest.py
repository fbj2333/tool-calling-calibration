"""Shared fixtures for ``tests/evaluation/`` test modules.

Pytest auto-discovers ``conftest.py`` and makes its fixtures available
to every test in the same directory tree. The helpers below back the
register-runtime-model tests in ``test_score.py`` and the BFCL
benchmark driver test in ``test_runner.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass
class _FakeModelConfig:
    model_name: str
    display_name: str
    url: str
    org: str
    license: str
    model_handler: object
    input_price: float | None = None
    output_price: float | None = None
    is_fc_model: bool = True
    underscore_to_dot: bool = False


@pytest.fixture
def fake_model_config() -> type[_FakeModelConfig]:
    """Stand-in for ``bfcl_eval``'s ``ModelConfig`` dataclass."""

    return _FakeModelConfig


@pytest.fixture
def make_model_dir():
    """Create a fake local model directory with a minimal ``config.json``.

    Returns a callable so individual tests can override ``model_type``,
    directory ``name``, and any extra config fields. Mirrors the
    on-disk layout that ``bfcl_calibration.evaluation.bfcl._register_runtime_model``
    inspects when picking which BFCL handler to register.
    """

    def _make(
        tmp_path: Path,
        *,
        model_type: str = "qwen3",
        name: str = "Qwen3-8B",
        config_extra: dict[str, object] | None = None,
    ) -> Path:
        model_dir = tmp_path / name
        model_dir.mkdir(parents=True)
        config = {"model_type": model_type}
        if config_extra:
            config.update(config_extra)
        (model_dir / "config.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        return model_dir

    return _make
