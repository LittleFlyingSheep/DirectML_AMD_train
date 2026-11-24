from copy import deepcopy
from io import BytesIO

import pytest

from Direct_AMD_train import enable, resolve_device, save_weights


@pytest.mark.parametrize("dataset,shape", [
    ("fashion-mnist", (1, 28, 28)),
    ("cifar10", (3, 32, 32)),
])
def test_tutorial_model_training_and_inference(dataset, shape):
    dml = pytest.importorskip("torch_directml")
    if not dml.is_available():
        pytest.skip("没有可用的 DirectML 设备")

    import torch
    from examples.torch_tutorials import build_model, make_optimizer

    enable(framework="torch", verbose=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cpu_model = build_model(dataset)
        images = torch.rand(8, *shape)
        labels = torch.arange(8)
    device = resolve_device()
    dml_model = deepcopy(cpu_model).to(device)
    cpu_opt, dml_opt = (make_optimizer(dataset, model) for model in (cpu_model, dml_model))

    for batch in (slice(None), slice(-1, None)):
        losses = []
        for model, optimizer, dev in ((cpu_model, cpu_opt, "cpu"), (dml_model, dml_opt, device)):
            optimizer.zero_grad()
            output = model(images[batch].to(dev))
            loss = torch.nn.functional.cross_entropy(output, labels[batch].to(dev))
            assert loss.device == torch.device(dev)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu())
        torch.testing.assert_close(*losses, atol=2e-5, rtol=2e-4)
        for ref, actual in zip(cpu_model.parameters(), dml_model.parameters()):
            torch.testing.assert_close(actual.grad.cpu(), ref.grad, atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(actual.detach().cpu(), ref.detach(), atol=2e-5, rtol=2e-4)

    checkpoint = BytesIO()
    save_weights(dml_model, checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    restored = build_model(dataset)
    restored.load_state_dict(state)
    assert next(dml_model.parameters()).device == device
    with torch.inference_mode():
        expected = cpu_model.eval()(images)
        actual = restored.to(device).eval()(images.to(device))
        assert actual.device == device
        assert not actual.requires_grad
        torch.testing.assert_close(actual.cpu(), dml_model.eval()(images.to(device)).cpu(),
                                   atol=0, rtol=0)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
