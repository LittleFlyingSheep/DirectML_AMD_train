import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from Direct_AMD_train import enable, export_onnx, patch_inference_mode, resolve_device


def test_inference_mode_preserves_decorator_and_explicit_disable():
    torch = pytest.importorskip("torch")
    patch_inference_mode()

    @torch.inference_mode
    def evaluate(x):
        return x * 2

    x = torch.ones(2, requires_grad=True)
    assert not evaluate(x).requires_grad
    with torch.inference_mode():
        assert not torch.is_inference_mode_enabled()
        assert not (x * 2).requires_grad
    with torch.inference_mode(False):
        assert (x * 2).requires_grad


def test_export_uses_returned_path(tmp_path, monkeypatch):
    (tmp_path / "best.onnx").write_bytes(b"old")
    exported = tmp_path / "custom.onnx"
    exported.write_bytes(b"new")
    model = SimpleNamespace(export=lambda **kwargs: str(exported))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda _: model))

    target = Path(export_onnx(str(tmp_path / "custom.pt"), str(tmp_path / "output")))
    assert target.read_bytes() == b"new"
    exported.unlink()
    with pytest.raises(FileNotFoundError):
        export_onnx(str(tmp_path / "custom.pt"), str(target.parent))


def test_torch_training_and_inference_match_cpu():
    dml = pytest.importorskip("torch_directml")
    if not dml.is_available():
        pytest.skip("没有可用的 DirectML 设备")

    import torch
    from torch import nn

    enable(framework="torch", verbose=False)
    device = resolve_device("privateuseone:0")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
        x = torch.randn(2, 3, 8, 8)
        y = torch.rand(2, 4, 6, 6)

    def step(net, dev):
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        loss = nn.functional.mse_loss(net(x.to(dev)), y.to(dev))
        loss.backward()
        grads = [p.grad.detach().cpu() for p in net.parameters()]
        optimizer.step()
        net.eval()
        with torch.inference_mode():
            prediction = net(x.to(dev))
        assert prediction.device == torch.device(dev)
        assert not prediction.requires_grad
        return [loss.detach().cpu(), *grads, *(p.detach().cpu() for p in net.parameters()),
                prediction.cpu()]

    dml_model = deepcopy(model).to(device)
    for actual, expected in zip(step(dml_model, device), step(model, "cpu")):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
