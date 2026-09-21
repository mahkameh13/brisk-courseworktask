#!/usr/bin/env python3
"""Assignment tests for Exercise 3 and Bonus 4. CPU-safe. 57 tests."""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bonus_resnet import (  # noqa: E402
    build_feature_transform,
    build_frozen_resnet,
    evaluate_features,
    extract_features,
    reduce_dimensions,
)
from exercise3_vae import (  # noqa: E402
    ConvVAE,
    interpolate,
    kl_divergence,
    sample_from_prior,
    vae_loss,
)

LATENT_DIM = 8
BATCH = 4

_TESTS = {}


def test(fn):
    _TESTS[fn.__name__] = fn
    return fn


def close(a, b, rel=1e-5, abs_tol=1e-8):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)


def run(prefix=""):
    names = sorted(n for n in _TESTS if n.startswith(prefix))
    if not names:
        print(f"no tests matching {prefix!r}")
        return False
    failures = []
    for name in names:
        try:
            _TESTS[name]()
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    passed = len(names) - len(failures)
    print(f"{passed}/{len(names)} passed")
    for name, err in failures:
        print(f"  FAIL  {name}")
        print(f"        {err}")
    return not failures


@test
def test_construct_records_latent_dim():
    assert ConvVAE(latent_dim=2).latent_dim == 2
    assert ConvVAE(latent_dim=128).latent_dim == 128


@test
def test_construct_has_trainable_parameters():
    model = ConvVAE(latent_dim=LATENT_DIM)
    assert any(p.requires_grad for p in model.parameters())


@test
def test_encode_returns_two_tensors():
    out = ConvVAE(LATENT_DIM).encode(torch.rand(BATCH, 1, 128, 128))
    assert isinstance(out, tuple) and len(out) == 2


@test
def test_encode_shapes():
    mu, logvar = ConvVAE(LATENT_DIM).encode(torch.rand(BATCH, 1, 128, 128))
    assert mu.shape == (BATCH, LATENT_DIM), mu.shape
    assert logvar.shape == (BATCH, LATENT_DIM), logvar.shape


@test
def test_encode_mu_and_logvar_are_distinct():
    mu, logvar = ConvVAE(LATENT_DIM).encode(torch.rand(BATCH, 1, 128, 128))
    assert not torch.allclose(mu, logvar)


@test
def test_encode_different_inputs_differ():
    model = ConvVAE(LATENT_DIM)
    a = model.encode(torch.zeros(1, 1, 128, 128))[0]
    b = model.encode(torch.ones(1, 1, 128, 128))[0]
    assert not torch.allclose(a, b)


@test
def test_reparam_shape_matches_mu():
    model = ConvVAE(LATENT_DIM)
    mu = torch.zeros(BATCH, LATENT_DIM)
    assert model.reparameterize(mu, torch.zeros_like(mu)).shape == mu.shape


@test
def test_reparam_is_stochastic():
    model = ConvVAE(LATENT_DIM)
    mu = torch.zeros(BATCH, LATENT_DIM)
    logvar = torch.zeros(BATCH, LATENT_DIM)
    assert not torch.allclose(
        model.reparameterize(mu, logvar), model.reparameterize(mu, logvar)
    )


@test
def test_reparam_collapses_to_mu_when_variance_vanishes():
    model = ConvVAE(LATENT_DIM)
    mu = torch.randn(BATCH, LATENT_DIM)
    logvar = torch.full((BATCH, LATENT_DIM), -50.0)
    assert torch.allclose(model.reparameterize(mu, logvar), mu, atol=1e-4)


@test
def test_reparam_empirical_moments():
    torch.manual_seed(0)
    model = ConvVAE(LATENT_DIM)
    n = 20000
    mu = torch.full((n, 1), 3.0)
    logvar = torch.full((n, 1), math.log(4.0))
    z = model.reparameterize(mu, logvar)
    assert close(z.mean(), 3.0, rel=0, abs_tol=0.05), z.mean().item()
    assert close(z.std(), 2.0, rel=0, abs_tol=0.05), z.std().item()


@test
def test_reparam_gradients_reach_both_arguments():
    model = ConvVAE(LATENT_DIM)
    mu = torch.zeros(BATCH, LATENT_DIM, requires_grad=True)
    logvar = torch.zeros(BATCH, LATENT_DIM, requires_grad=True)
    model.reparameterize(mu, logvar).sum().backward()
    assert mu.grad is not None, "no gradient on mu"
    assert logvar.grad is not None, "no gradient on logvar"
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(logvar.grad).all()


@test
def test_reparam_preserves_dtype_and_device():
    model = ConvVAE(LATENT_DIM)
    mu = torch.zeros(BATCH, LATENT_DIM, dtype=torch.float64)
    z = model.reparameterize(mu, torch.zeros_like(mu))
    assert z.dtype == mu.dtype
    assert z.device == mu.device


@test
def test_decode_shape():
    model = ConvVAE(LATENT_DIM)
    assert model.decode(torch.randn(BATCH, LATENT_DIM)).shape == (BATCH, 1, 128, 128)


@test
def test_decode_output_is_bounded():
    out = ConvVAE(LATENT_DIM).decode(torch.randn(BATCH, LATENT_DIM) * 10)
    assert out.min() >= 0.0 and out.max() <= 1.0


@test
def test_decode_accepts_a_prior_sample():
    out = ConvVAE(LATENT_DIM).decode(torch.randn(1, LATENT_DIM))
    assert out.shape == (1, 1, 128, 128)
    assert torch.isfinite(out).all()


@test
def test_forward_returns_three_values():
    out = ConvVAE(LATENT_DIM)(torch.rand(BATCH, 1, 128, 128))
    assert isinstance(out, tuple) and len(out) == 3


@test
def test_forward_shapes():
    x = torch.rand(BATCH, 1, 128, 128)
    x_hat, mu, logvar = ConvVAE(LATENT_DIM)(x)
    assert x_hat.shape == x.shape
    assert mu.shape == (BATCH, LATENT_DIM)
    assert logvar.shape == (BATCH, LATENT_DIM)


@test
def test_forward_reconstruction_is_bounded():
    x_hat, _, _ = ConvVAE(LATENT_DIM)(torch.rand(BATCH, 1, 128, 128))
    assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


@test
def test_forward_gradients_reach_the_encoder():
    model = ConvVAE(LATENT_DIM)
    model(torch.rand(BATCH, 1, 128, 128))[0].sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


@test
def test_kl_returns_one_value_per_sample():
    out = kl_divergence(torch.zeros(BATCH, LATENT_DIM), torch.zeros(BATCH, LATENT_DIM))
    assert out.shape == (BATCH,), out.shape


@test
def test_kl_zero_at_the_prior():
    out = kl_divergence(torch.zeros(BATCH, LATENT_DIM), torch.zeros(BATCH, LATENT_DIM))
    assert torch.allclose(out, torch.zeros(BATCH), atol=1e-6), out


@test
def test_kl_never_negative():
    torch.manual_seed(0)
    out = kl_divergence(torch.randn(64, LATENT_DIM) * 3, torch.randn(64, LATENT_DIM) * 2)
    assert (out >= -1e-6).all(), out.min().item()


@test
def test_kl_shifted_mean_hand_computed():
    out = kl_divergence(torch.tensor([[1.0]]), torch.tensor([[0.0]]))
    assert close(out.item(), 0.5, rel=0, abs_tol=1e-5), out.item()


@test
def test_kl_widened_variance_hand_computed():
    out = kl_divergence(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
    assert close(out.item(), 0.5 * (math.e - 2.0), rel=0, abs_tol=1e-5), out.item()


@test
def test_kl_narrowing_is_penalized():
    mu = torch.zeros(1, 1)
    wide = kl_divergence(mu, torch.tensor([[0.0]])).item()
    narrow = kl_divergence(mu, torch.tensor([[-4.0]])).item()
    assert narrow > wide, (wide, narrow)


@test
def test_kl_sums_over_latent_dimensions():
    one = kl_divergence(torch.tensor([[1.0]]), torch.tensor([[0.0]])).item()
    four = kl_divergence(torch.ones(1, 4), torch.zeros(1, 4)).item()
    assert close(four, 4 * one, rel=0, abs_tol=1e-5), (one, four)


@test
def test_kl_gradients_flow():
    mu = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
    logvar = torch.randn(BATCH, LATENT_DIM, requires_grad=True)
    kl_divergence(mu, logvar).sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def _loss_parts():
    torch.manual_seed(0)
    return (
        torch.rand(BATCH, 1, 128, 128),
        torch.rand(BATCH, 1, 128, 128),
        torch.randn(BATCH, LATENT_DIM),
        torch.randn(BATCH, LATENT_DIM),
    )


@test
def test_loss_returns_three_scalars():
    for value in vae_loss(*_loss_parts()):
        assert value.dim() == 0, value.shape


@test
def test_loss_total_is_the_weighted_sum():
    total, recon, kl = vae_loss(*_loss_parts(), beta=2.5)
    assert close(total.item(), recon.item() + 2.5 * kl.item())


@test
def test_loss_beta_zero_drops_the_kl_term():
    total, recon, _ = vae_loss(*_loss_parts(), beta=0.0)
    assert close(total.item(), recon.item())


@test
def test_loss_perfect_reconstruction_is_free():
    x = torch.rand(BATCH, 1, 128, 128)
    mu = torch.zeros(BATCH, LATENT_DIM)
    _, recon, _ = vae_loss(x, x, mu, torch.zeros_like(mu))
    assert close(recon.item(), 0.0, rel=0, abs_tol=1e-6), recon.item()


@test
def test_loss_sums_over_pixels():
    mu = torch.zeros(1, LATENT_DIM)
    logvar = torch.zeros(1, LATENT_DIM)
    big = vae_loss(
        torch.full((1, 1, 128, 128), 0.1), torch.zeros(1, 1, 128, 128), mu, logvar
    )[1]
    small = vae_loss(
        torch.full((1, 1, 64, 64), 0.1), torch.zeros(1, 1, 64, 64), mu, logvar
    )[1]
    assert close(big.item(), 4 * small.item(), rel=1e-4), (big.item(), small.item())


@test
def test_loss_invariant_to_batch_size():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 128, 128)
    x_hat = torch.rand(1, 1, 128, 128)
    mu = torch.randn(1, LATENT_DIM)
    logvar = torch.randn(1, LATENT_DIM)
    single = vae_loss(x_hat, x, mu, logvar)[0].item()
    doubled = vae_loss(
        x_hat.repeat(2, 1, 1, 1),
        x.repeat(2, 1, 1, 1),
        mu.repeat(2, 1),
        logvar.repeat(2, 1),
    )[0].item()
    assert close(doubled, single), (single, doubled)


@test
def test_loss_gradients_reach_the_model():
    model = ConvVAE(LATENT_DIM)
    x = torch.rand(BATCH, 1, 128, 128)
    x_hat, mu, logvar = model(x)
    vae_loss(x_hat, x, mu, logvar)[0].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )


@test
def test_end_to_end_loss_decreases():
    torch.manual_seed(0)
    model = ConvVAE(latent_dim=LATENT_DIM)
    x = torch.rand(BATCH, 1, 128, 128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    total = None
    for step in range(30):
        x_hat, mu, logvar = model(x)
        total, _, _ = vae_loss(x_hat, x, mu, logvar, beta=1.0)
        opt.zero_grad()
        total.backward()
        opt.step()
        if step == 0:
            first = total.item()
    assert total.item() < first, (first, total.item())


@test
def test_interpolate_shape_and_reproducible_means():
    torch.manual_seed(0)
    model = ConvVAE(LATENT_DIM).eval()
    xa = torch.rand(1, 1, 128, 128)
    xb = torch.rand(1, 1, 128, 128)
    frames = interpolate(model, xa, xb, steps=9)
    assert frames.shape == (9, 1, 128, 128)
    again = interpolate(model, xa, xb, steps=9)
    assert torch.allclose(frames, again, atol=1e-5)


@test
def test_prior_sample_shape():
    out = sample_from_prior(ConvVAE(LATENT_DIM).eval(), n=8)
    assert out.shape == (8, 1, 128, 128)
    assert out.min() >= 0.0 and out.max() <= 1.0


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _gray_image(value=128):
    return Image.fromarray(np.full((128, 128), value, dtype=np.uint8), mode="L")


@test
def test_transform_output_shape():
    assert build_feature_transform()(_gray_image()).shape == (3, 224, 224)


@test
def test_transform_size_is_configurable():
    assert build_feature_transform(size=128)(_gray_image()).shape == (3, 128, 128)


@test
def test_transform_replicates_grayscale_across_channels():
    out = build_feature_transform()(_gray_image())
    restored = torch.stack(
        [out[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c] for c in range(3)]
    )
    assert torch.allclose(restored[0], restored[1], atol=1e-5)
    assert torch.allclose(restored[1], restored[2], atol=1e-5)


@test
def test_transform_applies_imagenet_normalization():
    out = build_feature_transform()(_gray_image(255))
    for channel in range(3):
        expected = (1.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert close(out[channel].mean(), expected, rel=0, abs_tol=1e-3), (
            channel,
            out[channel].mean().item(),
            expected,
        )


def _resnet_available():
    return os.environ.get("SKIP_RESNET_WEIGHTS") != "1"


@test
def test_resnet_no_parameter_requires_gradient():
    if not _resnet_available():
        return
    assert all(not p.requires_grad for p in build_frozen_resnet().parameters())


@test
def test_resnet_is_in_evaluation_mode():
    if not _resnet_available():
        return
    assert not build_frozen_resnet().training


@test
def test_resnet_output_width_is_512():
    if not _resnet_available():
        return
    out = build_frozen_resnet()(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 512), out.shape


@test
def test_resnet_classifier_head_is_gone():
    if not _resnet_available():
        return
    assert build_frozen_resnet()(torch.randn(1, 3, 224, 224)).shape[1] != 1000


@test
def test_resnet_weights_are_pretrained_not_random():
    if not _resnet_available():
        return
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(build_frozen_resnet()(x), build_frozen_resnet()(x), atol=1e-5)


@test
def test_resnet_different_inputs_give_different_features():
    if not _resnet_available():
        return
    model = build_frozen_resnet()
    a = model(torch.zeros(1, 3, 224, 224))
    b = model(torch.ones(1, 3, 224, 224))
    assert not torch.allclose(a, b)


class _Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_was_enabled = None

    def forward(self, x):
        self.grad_was_enabled = torch.is_grad_enabled()
        return x.flatten(1)[:, :512]


def _stub_loader():
    images = torch.randn(10, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


@test
def test_extract_shapes_and_types():
    features, labels = extract_features(_Probe().eval(), _stub_loader())
    assert features.shape == (10, 512), features.shape
    assert labels.shape == (10,), labels.shape
    assert isinstance(features, np.ndarray) and isinstance(labels, np.ndarray)


@test
def test_extract_preserves_label_order():
    _, labels = extract_features(_Probe().eval(), _stub_loader())
    assert list(labels) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1], list(labels)


@test
def test_extract_builds_no_graph():
    probe = _Probe().eval()
    extract_features(probe, _stub_loader())
    assert probe.grad_was_enabled is not None, "extractor was never called"
    assert probe.grad_was_enabled is False, "autograd was on during extraction"


@test
def test_reduce_output_shapes():
    tr, va = reduce_dimensions(
        np.random.randn(137, 512), np.random.randn(41, 512), n_components=64
    )
    assert tr.shape == (137, 64), tr.shape
    assert va.shape == (41, 64), va.shape


@test
def test_reduce_structure_survives():
    rng = np.random.default_rng(0)
    stacked = np.vstack([rng.normal(0, 1, (100, 512)), rng.normal(20, 1, (100, 512))])
    reduced, _ = reduce_dimensions(stacked, stacked[:5], 8)
    gap = np.linalg.norm(reduced[:100].mean(0) - reduced[100:].mean(0))
    spread = reduced[:100].std(0).mean()
    assert gap > 5 * spread, (gap, spread)


@test
def test_reduce_shared_projection():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(80, 512))
    val = np.vstack([train[3], rng.normal(size=(19, 512))])
    train_reduced, val_reduced = reduce_dimensions(train, val, n_components=16)
    assert np.allclose(train_reduced[3], val_reduced[0], atol=1e-6), (
        "row present in both sets landed at two different points: "
        "the projection was not shared"
    )


@test
def test_reduce_validation_is_not_refitted():
    rng = np.random.default_rng(2)
    train = rng.normal(size=(80, 512))
    val = rng.normal(size=(20, 512))
    _, plain = reduce_dimensions(train, val, n_components=16)
    _, shifted = reduce_dimensions(train, val + 50.0, n_components=16)
    moved = np.linalg.norm(shifted.mean(0) - plain.mean(0))
    assert moved > 1.0, (
        f"shifting the validation set moved its projection by only {moved:.4f}; "
        "the projection appears to be refitted on the validation data"
    )


def _separable():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [30, 0], [0, 30], [30, 30]], dtype=float)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for label, center in enumerate(centers):
        tr_x.append(rng.normal(center, 1.0, (50, 2)))
        tr_y += [label] * 50
        va_x.append(rng.normal(center, 1.0, (20, 2)))
        va_y += [label] * 20
    return np.vstack(tr_x), np.array(tr_y), np.vstack(va_x), np.array(va_y)


@test
def test_evaluate_returns_expected_keys():
    assert set(evaluate_features(*_separable())) == {"accuracy", "silhouette", "report"}


@test
def test_evaluate_separable_data_scores_high():
    assert evaluate_features(*_separable())["accuracy"] > 0.95


@test
def test_evaluate_separable_data_has_high_silhouette():
    assert evaluate_features(*_separable())["silhouette"] > 0.7


@test
def test_evaluate_random_features_score_near_chance():
    rng = np.random.default_rng(0)
    result = evaluate_features(
        rng.normal(size=(400, 16)),
        np.tile([0, 1, 2, 3], 100),
        rng.normal(size=(200, 16)),
        np.tile([0, 1, 2, 3], 50),
    )
    assert result["accuracy"] < 0.40, result["accuracy"]


@test
def test_evaluate_neighbor_count_is_honored():
    tr_x, tr_y, va_x, va_y = _separable()
    assert evaluate_features(tr_x, tr_y, va_x, va_y, n_neighbors=1)["accuracy"] > 0.9


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "test_"
    ok = run(prefix)
    raise SystemExit(0 if ok else 1)
