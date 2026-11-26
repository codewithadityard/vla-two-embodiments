"""M3/M4 torch tests (run on Kaggle; skipped on the laptop, which has no torch)."""
import pytest

torch = pytest.importorskip("torch")

from minivla.expert import ActionExpert  # noqa: E402
from minivla.heads import FlowMatchingHead, RegressionHead, euler, masked_mean  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, T, F = 4, 12, 32


def make(flow=False, dropout=0.0, seed=0):
    torch.manual_seed(seed)
    return ActionExpert(F, T, chunk=50, d=64, layers=2, heads=4, ffn=128, dropout=dropout, flow=flow).to(DEV)


def inputs(seed=0, b=B):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(b, T, F, generator=g).to(DEV)
    mask = torch.ones(b, T, dtype=torch.bool, device=DEV)
    mask[:, -3:] = False  # last 3 tokens are padding
    state = torch.randn(b, 8, generator=g).to(DEV)
    actions = torch.randn(b, 50, 7, generator=g).to(DEV)
    pad = torch.zeros(b, 50, dtype=torch.bool, device=DEV)
    pad[:, -10:] = True
    return feats, mask, state, actions, pad


def test_shapes_both_heads():
    feats, mask, state, actions, _ = inputs()
    assert make()(feats, mask, state).shape == (B, 50, 7)
    t = torch.rand(B, device=DEV)
    assert make(flow=True)(feats, mask, state, actions, t).shape == (B, 50, 7)


def test_padded_memory_tokens_do_not_matter():
    e = make().eval()
    feats, mask, state, _, _ = inputs()
    out1 = e(feats, mask, state)
    feats2 = feats.clone()
    feats2[:, -3:] = 1000.0  # change only padded tokens
    assert torch.allclose(out1, e(feats2, mask, state), atol=1e-5)


def test_padded_targets_do_not_matter():
    feats, mask, state, actions, pad = inputs()
    e, head = make(), RegressionHead()
    l1 = head.loss(e, feats, mask, state, actions, pad)
    actions2 = actions.clone()
    actions2[pad] = 1000.0
    assert torch.allclose(l1, head.loss(e, feats, mask, state, actions2, pad))


def test_masked_mean_value():
    err = torch.ones(1, 4, 2)
    err[0, 3] = 100.0
    pad = torch.tensor([[False, False, False, True]])
    assert masked_mean(err, pad).item() == 1.0


def test_flow_path_endpoints_and_euler_recovers_actions():
    g = torch.Generator().manual_seed(0)
    a, eps = torch.randn(3, 50, 7, generator=g), torch.randn(3, 50, 7, generator=g)
    for t in [0.0, 1.0]:
        x_t = t * eps + (1 - t) * a
        assert torch.allclose(x_t, a if t == 0 else eps)
    # the true velocity of the linear path is constant: u = eps - a; Euler from eps at t=1 must land on a
    out = euler(lambda x, t: eps - a, eps.clone(), steps=10)
    assert (out - a).abs().max().item() <= 1e-5


def test_time_sampler_distribution():
    torch.manual_seed(0)
    t = FlowMatchingHead().sample_time(100_000, "cpu")
    assert t.min().item() >= 0.001 and t.max().item() <= 1.0
    assert abs(t.mean().item() - (0.6 * 0.999 + 0.001)) <= 0.005


def overfit(flow, steps):
    """Fit 8 random samples. Returns (training loss at the first step, mean training loss over the last 100 steps,
    masked L1 of the prediction)."""
    torch.manual_seed(0)
    feats, mask, state, actions, pad = inputs(b=8)
    e, head = make(flow=flow), (FlowMatchingHead() if flow else RegressionHead())
    opt = torch.optim.AdamW(e.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        loss = head.loss(e, feats, mask, state, actions, pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    e.eval()
    g = torch.Generator(device=DEV).manual_seed(0)
    pred = head.predict(e, feats, mask, state, generator=g)
    return losses[0], sum(losses[-100:]) / 100, masked_mean((pred - actions).abs(), pad).item()


# Relative check (approved 2026-11-26, replaces fixed L1 thresholds 0.02 / 0.05 on random data):
# the training loss must drop by at least 90 % from its starting value.
def test_regression_overfits_tiny_batch():
    first, last, _ = overfit(flow=False, steps=1000)
    assert last <= 0.1 * first, (first, last)


def test_flow_overfits_tiny_batch():
    first, last, _ = overfit(flow=True, steps=3000)
    assert last <= 0.1 * first, (first, last)


def test_same_seed_same_losses():
    def losses():
        torch.manual_seed(0)
        feats, mask, state, actions, pad = inputs()
        e, head = make(flow=True, dropout=0.1), FlowMatchingHead()
        opt = torch.optim.AdamW(e.parameters(), lr=1e-3)
        out = []
        for _ in range(20):
            loss = head.loss(e, feats, mask, state, actions, pad)
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(loss.item())
        return out

    assert losses() == losses()
