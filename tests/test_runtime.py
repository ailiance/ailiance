import pytest

pytest.importorskip("mlx")  # src.worker.runtime imports mlx; skip on CI/linux


def test_runtime_config_defaults():
    from src.worker.runtime import WorkerConfig

    cfg = WorkerConfig(
        model_path="/tmp/fake-model",
        adapters_dir="/tmp/fake-adapters",
        domains=["python", "rust"],
    )
    assert cfg.model_path == "/tmp/fake-model"
    assert cfg.port == 9201
    assert cfg.precision == "bf16"


def test_runtime_lora_switch_interface():
    """Runtime exposes apply(domain) and generate() interface."""
    from src.worker.runtime import MLXWorkerRuntime

    assert hasattr(MLXWorkerRuntime, "apply")
    assert hasattr(MLXWorkerRuntime, "generate")
    assert hasattr(MLXWorkerRuntime, "preload_adapters")
