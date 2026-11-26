"""Action expert: a small transformer decoder that turns backbone tokens + state into an action chunk.

Memory (keys/values for cross-attention): backbone tokens -> LayerNorm + Linear(960 -> d), plus one
state token Linear(8 -> d), plus a learned embedding per memory position (the token layout is fixed,
so each position always means the same camera patch or text slot).
Queries: one per action step (50), with a learned position embedding. The flow head also feeds the
noisy action x_t and the time t into the queries, as SmolVLA does (action_time_mlp).
Self-attention between action tokens is causal, as in SmolVLA.
"""
import math

import torch
import torch.nn as nn


def sinusoidal(t, dim, min_period=4e-3, max_period=4.0):
    """Sine-cosine embedding of t in [0, 1] -> [B, dim], periods as in SmolVLA (min 0.004, max 4.0)."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angle = (2 * math.pi / period)[None, :] * t.float()[:, None]
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class ActionExpert(nn.Module):
    def __init__(self, feat_dim, n_feat_tokens, state_dim=8, act_dim=7, chunk=50,
                 d=512, layers=6, heads=8, ffn=2048, dropout=0.1, flow=False):
        super().__init__()
        self.chunk, self.flow, self.d = chunk, flow, d
        self.feat_in = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
        self.state_in = nn.Linear(state_dim, d)
        self.mem_pos = nn.Parameter(torch.randn(1, n_feat_tokens + 1, d) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, chunk, d) * 0.02)
        if flow:
            self.act_in = nn.Linear(act_dim, d)
            self.time_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerDecoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, act_dim)
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(chunk), persistent=False)

    def forward(self, feats, feat_mask, state, x_t=None, t=None):
        """feats [B, T, F], feat_mask [B, T] (True = real token), state [B, S]; flow: x_t [B, chunk, A], t [B].
        Returns [B, chunk, A] (actions for regression, velocity for flow)."""
        b = feats.shape[0]
        memory = torch.cat([self.feat_in(feats), self.state_in(state)[:, None]], dim=1) + self.mem_pos
        memory_pad = ~torch.cat([feat_mask, feat_mask.new_ones(b, 1)], dim=1)
        q = self.query_pos.expand(b, -1, -1)
        if self.flow:
            time = sinusoidal(t, self.d).to(q.dtype)[:, None].expand(-1, self.chunk, -1)
            q = q + self.time_mlp(torch.cat([self.act_in(x_t), time], dim=-1))
        h = self.decoder(q, memory, tgt_mask=self.causal, tgt_is_causal=True, memory_key_padding_mask=memory_pad)
        return self.out(h)
