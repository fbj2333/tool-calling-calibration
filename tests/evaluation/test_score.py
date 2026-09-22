from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import bfcl_calibration.evaluation.bfcl._gptoss_handlers as _gptoss_handlers_mod
import bfcl_calibration.evaluation.bfcl._patches as _patches_mod
import bfcl_calibration.evaluation.bfcl._sri_loop as _sri_loop_mod
from bfcl_calibration.evaluation import bfcl


def test_register_runtime_model_adds_qwen3_fc_config(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(tmp_path)
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        QwenFCHandler=object,
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    expected_hash = hashlib.sha1(str(model_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    assert runtime_name == f"bfcl_calibration-local-Qwen3-8B-{expected_hash}-fc"
    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_name == str(model_dir)
    assert config.model_handler is object
    assert config.is_fc_model is True
    # Upstream scorer converts result-dir underscores back to slashes before
    # looking up MODEL_CONFIG_MAPPING, so dynamic local names need both keys.
    assert handles.MODEL_CONFIG_MAPPING[runtime_name.replace("_", "/")] is config


def test_runtime_model_name_distinguishes_same_basename_different_paths(
    tmp_path: Path, make_model_dir
) -> None:
    model_dir_a = make_model_dir(tmp_path / "a", name="checkpoint-1000")
    model_dir_b = make_model_dir(tmp_path / "b", name="checkpoint-1000")

    name_a = bfcl._runtime_model_name(model_dir_a)
    name_b = bfcl._runtime_model_name(model_dir_b)

    assert name_a != name_b
    assert name_a.startswith("bfcl_calibration-local-checkpoint-1000-")
    assert name_b.startswith("bfcl_calibration-local-checkpoint-1000-")


def test_register_runtime_model_adds_llama31_fc_config(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(tmp_path, model_type="llama")
    llama_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        Llama31Handler=llama_handler,
        SalesforceLlamaHandler=object(),
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is llama_handler
    assert config.is_fc_model is True


def test_register_runtime_model_adds_salesforce_llama_fc_config(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(
        tmp_path,
        model_type="llama",
        name="Llama-xLAM-2-8b-fc-r",
        config_extra={"_name_or_path": "Salesforce/Llama-xLAM-2-8b-fc-r"},
    )
    salesforce_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        Llama31Handler=object(),
        SalesforceLlamaHandler=salesforce_handler,
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is salesforce_handler
    assert config.is_fc_model is True


@pytest.mark.parametrize(
    ("model_name", "hf_name"),
    [
        ("xLAM-2-1b-fc-r", "Salesforce/xLAM-2-1b-fc-r"),
        ("xLAM-2-3b-fc-r", "Salesforce/xLAM-2-3b-fc-r"),
        ("xLAM-2-32b-fc-r", "Salesforce/xLAM-2-32b-fc-r"),
    ],
)
def test_register_runtime_model_routes_salesforce_qwen_xlam_fc_config(
    tmp_path: Path,
    make_model_dir,
    fake_model_config,
    model_name: str,
    hf_name: str,
) -> None:
    model_dir = make_model_dir(
        tmp_path,
        model_type="qwen2",
        name=model_name,
        config_extra={"_name_or_path": hf_name},
    )
    salesforce_qwen_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        SalesforceLlamaHandler=object(),
        SalesforceQwenHandler=salesforce_qwen_handler,
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is salesforce_qwen_handler
    assert config.is_fc_model is True


def test_register_runtime_model_routes_salesforce_70b_xlam_to_llama_handler(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(
        tmp_path,
        model_type="llama",
        name="Llama-xLAM-2-70b-fc-r",
        config_extra={"_name_or_path": "Salesforce/Llama-xLAM-2-70b-fc-r"},
    )
    salesforce_llama_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        SalesforceLlamaHandler=salesforce_llama_handler,
        SalesforceQwenHandler=object(),
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is salesforce_llama_handler
    assert config.is_fc_model is True


def test_register_runtime_model_rejects_unknown_salesforce_xlam_model_type(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(
        tmp_path,
        model_type="mistral",
        name="xLAM-2-experimental-fc-r",
        config_extra={"_name_or_path": "Salesforce/xLAM-2-experimental-fc-r"},
    )
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        SalesforceLlamaHandler=object(),
        SalesforceQwenHandler=object(),
    )

    with pytest.raises(ValueError, match="unsupported config.json"):
        bfcl._register_runtime_model(handles, model_path=model_dir)


def test_register_runtime_model_rejects_unsupported_model_type(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(tmp_path, model_type="mpt")
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        QwenFCHandler=object,
    )

    with pytest.raises(ValueError, match="Local model not recognised"):
        bfcl._register_runtime_model(handles, model_path=model_dir)


def test_register_runtime_model_routes_hammer_by_path_keyword(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    # Hammer is Qwen2-based but needs upstream's HammerHandler; path keyword
    # must override the generic model_type fall-through (which would otherwise
    # reject qwen2).
    model_dir = make_model_dir(tmp_path, model_type="qwen2", name="Hammer2.1-1.5b")
    hammer_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        HammerHandler=hammer_handler,
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is hammer_handler
    assert config.is_fc_model is True


def test_register_runtime_model_routes_toolace_by_path_keyword(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    # ToolACE-2-8B is Llama-based but upstream uses LlamaHandler (not
    # LlamaHandler_3_1) and marks ``is_fc_model=False``.
    model_dir = make_model_dir(tmp_path, model_type="llama", name="ToolACE-2-8B")
    llama_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        LlamaHandler=llama_handler,
        Llama31Handler=object(),
        SalesforceLlamaHandler=object(),
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is llama_handler
    assert config.is_fc_model is False


def test_register_runtime_model_routes_watt_tool_by_path_keyword(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(tmp_path, model_type="llama", name="watt-tool-8B")
    llama_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        LlamaHandler=llama_handler,
        Llama31Handler=object(),
        SalesforceLlamaHandler=object(),
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is llama_handler
    assert config.is_fc_model is False


def test_register_runtime_model_routes_granite_3_by_path_keyword(
    tmp_path: Path, make_model_dir, fake_model_config
) -> None:
    model_dir = make_model_dir(
        tmp_path, model_type="granite", name="granite-3.2-8b-instruct"
    )
    granite_3_handler = object()
    handles = SimpleNamespace(
        ModelConfig=fake_model_config,
        MODEL_CONFIG_MAPPING={},
        Granite3FCHandler=granite_3_handler,
    )

    runtime_name = bfcl._register_runtime_model(handles, model_path=model_dir)

    config = handles.MODEL_CONFIG_MAPPING[runtime_name]
    assert config.model_handler is granite_3_handler
    assert config.is_fc_model is True


def test_load_bfcl_handles_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    vendor_root = tmp_path / "vendor"
    vendor_root.mkdir(parents=True)
    monkeypatch.setattr(bfcl, "_ensure_bfcl_vendor_root", lambda: vendor_root)

    imported: list[str] = []

    def _fake_import_module(name: str):
        imported.append(name)
        if name == "bfcl_eval.constants.model_config":
            raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
        return SimpleNamespace()

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    with pytest.raises(RuntimeError, match="anthropic"):
        bfcl._load_bfcl_handles(tmp_path / "project_root")

    assert imported[:3] == [
        "bfcl_eval._llm_response_generation",
        "bfcl_eval.eval_checker.eval_runner",
        "bfcl_eval.constants.model_config",
    ]


def test_preflight_backend_requirements_reports_missing_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace() if name == "transformers" else original_import_module(name),
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[vllm\]\""):
        bfcl._preflight_backend_requirements()


class _AutoAttr(SimpleNamespace):
    """Vendor-module stand-in that yields a fresh sentinel for any attr.

    ``_load_bfcl_handles`` reads many attributes off the imported vendor
    modules (``.main`` / handler classes / ``.OSSHandler`` / ...). A bare
    ``SimpleNamespace`` raises on unknown attrs; this returns a stable
    per-name object so the loader can finish building ``BFCLOfficialHandles``
    without a real ``bfcl_eval`` tree.
    """

    def __getattr__(self, name: str):  # only called on a miss
        obj = object()
        setattr(self, name, obj)
        return obj


def test_load_bfcl_handles_installs_vllm_args_gptoss_handler_and_probe_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The patch installers are replaced on their source submodules, which is
    # where the loader imports them from at call time.
    vendor_root = tmp_path / "vendor"
    vendor_root.mkdir(parents=True)
    monkeypatch.setattr(bfcl, "_ensure_bfcl_vendor_root", lambda: vendor_root)
    monkeypatch.setattr(importlib, "import_module", lambda name: _AutoAttr())

    calls: dict[str, int] = {"vllm_args": 0, "gptoss": 0, "sri": 0}

    def _count(key, result=None):
        def _fn(*args, **kwargs):
            calls[key] += 1
            return result
        return _fn

    gptoss_handler = object()
    monkeypatch.setattr(_patches_mod, "_patch_vllm_server_args", _count("vllm_args"))
    monkeypatch.setattr(_gptoss_handlers_mod, "_make_gptoss_fc_handler", _count("gptoss", gptoss_handler))
    monkeypatch.setattr(_sri_loop_mod, "_apply_sri_patches", _count("sri"))

    handles = bfcl._load_bfcl_handles(tmp_path / "project_root")

    assert calls == {"vllm_args": 1, "gptoss": 1, "sri": 1}
    assert handles.GPTOSSFCHandler is gptoss_handler


def test_apply_sri_patches_inert_when_sri_and_cri_unset() -> None:
    # The real ``_apply_sri_patches`` returns immediately when
    # BFCL_SRI_MODE=off and CRI off (it never installs the sentinel).
    # Exercise it directly with a fake BaseHandler module so no monkeypatch
    # masks the early-return; assert no ``_bfclcalib_sri_patched`` attr.
    import os

    saved = {
        k: os.environ.pop(k, None)
        for k in ("BFCL_SRI_MODE", "BFCL_CRI_MODE", "BFCL_SRI_VARIANT", "BFCL_CRI_VARIANT")
    }
    try:

        class _FC:
            pass

        class _PR:
            pass

        class _BaseHandler:
            inference_multi_turn_FC = _FC
            inference_multi_turn_prompting = _PR

        base_handler_module = SimpleNamespace(BaseHandler=_BaseHandler)

        _sri_loop_mod._apply_sri_patches(base_handler_module)

        assert not getattr(_BaseHandler.inference_multi_turn_FC, "_bfclcalib_sri_patched", False)
        assert not getattr(
            _BaseHandler.inference_multi_turn_prompting, "_bfclcalib_sri_patched", False
        )
        assert _BaseHandler.inference_multi_turn_FC is _FC
        assert _BaseHandler.inference_multi_turn_prompting is _PR
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
