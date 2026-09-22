from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bfcl_calibration.evaluation import bfcl


def test_run_bfcl_benchmark_uses_partial_eval_and_writes_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_model_dir,
    fake_model_config,
) -> None:
    model_dir = make_model_dir(tmp_path)
    calls: dict[str, object] = {}
    for name in ("BFCL_SRI_MODE", "BFCL_CRI_MODE", "BFCL_SRI_LOG_PATH", "BFCL_SEED"):
        monkeypatch.delenv(name, raising=False)

    def _fake_generation_main(args) -> None:
        calls["generation"] = args

    def _fake_evaluation_main(model, test_categories, result_dir, score_dir, partial_eval=False) -> None:
        calls["evaluation"] = {
            "model": model,
            "test_categories": test_categories,
            "result_dir": result_dir,
            "score_dir": score_dir,
            "partial_eval": partial_eval,
        }
        project_root = Path(os.environ["BFCL_PROJECT_ROOT"])
        model_dirname = model[0].replace("/", "_")
        score_root = project_root / score_dir / model_dirname / "multi_turn"
        score_root.mkdir(parents=True, exist_ok=True)
        (score_root / "BFCL_v4_multi_turn_base_score.json").write_text(
            "\n".join(
                [
                    json.dumps({"accuracy": 0.5, "correct_count": 1, "total_count": 2}),
                    json.dumps({"error_type": "multi_turn:force_terminated"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _fake_load_bfcl_handles(project_root: Path):
        monkeypatch.setenv("BFCL_PROJECT_ROOT", str(project_root))
        return SimpleNamespace(
            generation_main=_fake_generation_main,
            evaluation_main=_fake_evaluation_main,
            ModelConfig=fake_model_config,
            MODEL_CONFIG_MAPPING={},
            QwenFCHandler=object,
            load_dataset_entry=lambda category: [
                {"id": f"{category}_0"},
                {"id": f"{category}_1"},
                {"id": f"{category}_2"},
            ],
            get_directory_structure_by_category=lambda category: "multi_turn",
        )

    monkeypatch.setattr(bfcl, "_load_bfcl_handles", _fake_load_bfcl_handles)
    monkeypatch.setattr(bfcl, "_preflight_backend_requirements", lambda: calls.setdefault("preflight", True))

    output_dir = tmp_path / "run"
    summary = bfcl.run_bfcl_benchmark(
        model_path=str(model_dir),
        categories=["multi_turn_base"],
        output_dir=str(output_dir),
        max_samples=2,
    )

    generation_args = calls["generation"]
    expected_hash = hashlib.sha1(str(model_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    assert generation_args.result_dir == "bfcl_results"
    assert generation_args.run_ids is True
    assert generation_args.local_model_path == str(model_dir)
    assert generation_args.model == [f"bfcl_calibration-local-Qwen3-8B-{expected_hash}-fc"]
    assert generation_args.backend == "vllm"

    assert calls["evaluation"] == {
        "model": [f"bfcl_calibration-local-Qwen3-8B-{expected_hash}-fc"],
        "test_categories": ["multi_turn_base"],
        "result_dir": "bfcl_results",
        "score_dir": "bfcl_scores",
        "partial_eval": True,
    }
    assert calls["preflight"] is True

    subset_file = output_dir / "test_case_ids_to_generate.json"
    assert json.loads(subset_file.read_text(encoding="utf-8")) == {
        "multi_turn_base": ["multi_turn_base_0", "multi_turn_base_1"]
    }

    summary_json = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_json["partial_eval"] is True
    assert summary_json["run_config"]["bfcl_seed"] == ""
    assert summary_json["run_config"]["bfcl_sri_mode"] == "off"
    assert summary_json["overall"]["accuracy"] == pytest.approx(0.5)
    assert summary_json["categories"][0]["error_type_breakdown"] == {
        "multi_turn:force_terminated": 1
    }
    assert summary["runtime_model_name"] == f"bfcl_calibration-local-Qwen3-8B-{expected_hash}-fc"
