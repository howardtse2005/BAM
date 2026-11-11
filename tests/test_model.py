import pytest
import torch

from training.unet import (
    BAMUNet,
    build_model,
    count_parameters,
    estimate_gflops,
    load_checkpoint_model,
)


def test_forward_shape():
    model = BAMUNet()
    out = model(torch.randn(2, 4, 256, 256))
    assert out.shape == (2, 1, 256, 256)


def test_parameter_budget():
    model = BAMUNet()
    assert count_parameters(model) < 1_000_000
    assert count_parameters(model) > 100_000


def test_odd_input_size_is_handled():
    model = BAMUNet().eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 4, 200, 296))
    assert out.shape == (1, 1, 200, 296)


def test_rejects_wrong_channels():
    model = BAMUNet()
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 64, 64))


def test_build_model_from_dict():
    model = build_model({"width": 16, "depth": 3, "block": "vanilla"})
    assert model.width == 16
    assert model.depth == 3
    assert model.block == "vanilla"
    assert model(torch.randn(1, 4, 64, 64)).shape == (1, 1, 64, 64)


def test_output_starts_low_confidence():
    model = BAMUNet().eval()
    with torch.inference_mode():
        out = model(torch.randn(1, 4, 64, 64))
    # The head bias starts at -2 so early training is stable and quiet.
    assert float(out.max()) < 0.0
    assert float(torch.sigmoid(out).max()) < 0.3


def test_gflops_is_reported():
    model = BAMUNet()
    gflops = estimate_gflops(model, size=128)
    assert 0.05 < gflops < 10.0


def test_checkpoint_roundtrip(tmp_path):
    model = BAMUNet(width=12, depth=3)
    path = tmp_path / "toy.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.config_dict(),
            "data_tier": "A",
            "epoch": 3,
        },
        path,
    )

    restored = load_checkpoint_model(path)
    assert restored.width == 12
    assert restored.depth == 3

    x = torch.randn(1, 4, 64, 64)
    with torch.inference_mode():
        assert torch.allclose(model.eval()(x), restored(x), atol=1e-5)


def test_training_step_reduces_loss():
    from training.losses import BAMLoss

    torch.manual_seed(0)
    model = BAMUNet(width=8, depth=2)
    criterion = BAMLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = torch.randn(2, 4, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 28:32, 10:54] = 1.0
    weight = torch.ones_like(target)

    first = None
    for _ in range(6):
        optimizer.zero_grad()
        loss, _ = criterion(model(x), target, weight)
        loss.backward()
        optimizer.step()
        first = float(loss.detach()) if first is None else first

    assert float(loss.detach()) < first
