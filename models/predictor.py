"""MLP and Transformer predictors f_theta: (z_win, a_win) -> z_{t+1}_hat."""
from __future__ import annotations
import torch
import torch.nn as nn


class MLPPredictor(nn.Module):
    """MLP predictor with n_layers hidden layers.

    window=1: input = [z_t, a_t]  (standard Markov)
    window>1: input = [z_{t-W+1},...,z_t, a_{t-W+1},...,a_t]  (pre-flattened by caller)

    n_layers : number of hidden layers (output layer is always added on top)
    activation: 'relu' | 'elu'
    """
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=256, n_layers=2, window=1,
                 activation='elu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.window = window
        act_fn = nn.ReLU() if activation == 'relu' else nn.ELU()
        in_dim = window * (latent_dim + action_dim)
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), act_fn]
        layers.append(nn.Linear(hidden_dim if n_layers > 0 else in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, a):
        # z: (B, W*d) or (B, d) when window=1
        # a: (B, W*d_a) or (B, d_a) when window=1
        return self.net(torch.cat([z, a], dim=-1))


# ── Transformer predictor ──────────────────────────────────────────────────────

class _PredTransformerBlock(nn.Module):
    """Pre-LN transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerPredictor(nn.Module):
    """Transformer predictor with DINO-WM-style action lifting.

    Drop-in replacement for MLPPredictor — identical call signature:
        forward(z, a)
            z : (B, W*latent_dim)   window of latents, flattened
            a : (B, W*action_dim)   window of encoded actions, flattened
        returns (B, latent_dim)

    Action lifting (following DINO-WM §3.1.2): state tokens and action
    embeddings are projected **separately** to embed_dim and **added**
    per timestep, so each token is conditioned on its action before
    self-attention. This differs from a joint [z‖a] projection.

        z_proj(z_t) + a_proj(a_t)  →  token_t  ∈ R^embed_dim
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 1,
        window: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.window     = window
        self.embed_dim  = embed_dim

        # Separate projections: state and action are lifted independently
        self.z_proj = nn.Linear(latent_dim, embed_dim)
        self.a_proj = nn.Linear(action_dim, embed_dim)   # action lifting

        self.pos_emb = nn.Parameter(torch.zeros(1, window, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList([
            _PredTransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        W = self.window
        z_seq = z.reshape(B, W, self.latent_dim)          # (B, W, d)
        a_seq = a.reshape(B, W, self.action_dim)          # (B, W, d_a)
        # Action lifting: add per-token action embedding to state token
        tokens = self.z_proj(z_seq) + self.a_proj(a_seq)  # (B, W, embed_dim)
        tokens = tokens + self.pos_emb
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, -1])                   # (B, latent_dim)
