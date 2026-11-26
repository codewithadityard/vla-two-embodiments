"""Two ways to train/use the same expert: L1 regression (config C) and flow matching (configs A, B).

Flow matching follows SmolVLA (modeling_smolvla.py at the pinned LeRobot commit):
noise at t = 1, x_t = t * eps + (1 - t) * a, target velocity u = eps - a, MSE loss,
t ~ Beta(1.5, 1) * 0.999 + 0.001, and Euler integration from t = 1 to t = 0 at inference.
Losses are averaged over real (non-padded) action steps only.
"""
import torch


def masked_mean(err, pad):
    """Mean of err [B, chunk, A] over steps where pad [B, chunk] is False."""
    keep = (~pad).unsqueeze(-1).to(err.dtype)
    return (err * keep).sum() / (keep.sum() * err.shape[-1])


class RegressionHead:
    def loss(self, expert, feats, mask, state, actions, pad):
        return masked_mean((expert(feats, mask, state).float() - actions).abs(), pad)

    @torch.no_grad()
    def predict(self, expert, feats, mask, state, generator=None):
        return expert(feats, mask, state).float()


class FlowMatchingHead:
    def __init__(self, steps=10, beta_a=1.5, beta_b=1.0):
        self.steps = steps
        self.beta = torch.distributions.Beta(beta_a, beta_b)

    def sample_time(self, n, device):
        return self.beta.sample((n,)).to(device) * 0.999 + 0.001

    def loss(self, expert, feats, mask, state, actions, pad, noise=None, t=None):
        """noise / t can be given (fixed) for diagnostics; by default they are drawn fresh."""
        noise = torch.randn_like(actions) if noise is None else noise
        t = self.sample_time(actions.shape[0], actions.device) if t is None else t
        tt = t[:, None, None]
        x_t = tt * noise + (1 - tt) * actions
        u = noise - actions
        v = expert(feats, mask, state, x_t, t).float()
        return masked_mean((v - u) ** 2, pad)

    @torch.no_grad()
    def predict(self, expert, feats, mask, state, generator=None, steps=None):
        """Euler from pure noise at t = 1 to t = 0 in `steps` steps (default: self.steps). generator fixes the noise."""
        b = feats.shape[0]
        x = torch.randn(b, expert.chunk, expert.out.out_features, device=feats.device, generator=generator)
        return euler(lambda x, t: expert(feats, mask, state, x, t).float(), x, steps or self.steps)


def euler(velocity, x, steps):
    """Integrate dx/dt = velocity(x, t) from t = 1 to t = 0 with `steps` equal Euler steps."""
    dt = -1.0 / steps
    t = torch.ones(x.shape[0], device=x.device)
    for _ in range(steps):
        x = x + dt * velocity(x, t)
        t = t + dt
    return x
