"""Sensorimotor world model adapted from Ivashkov et al. for CartPole."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchViTEncoder(nn.Module):
    """ViT-Tiny-style encoder with CLS pooling and an MLP projector."""

    def __init__(self, image_size=128, patch_size=8, in_channels=3,
                 proprio_dim=0,
                 embed_dim=192, depth=4, heads=3, mlp_ratio=4.0,
                 latent_dim=192):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        n = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, int(embed_dim * mlp_ratio), batch_first=True,
            norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.proprio_dim = int(proprio_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim + self.proprio_dim, 2048),
            nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, latent_dim))
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, pixels, proprio=None):
        x = self.patch(pixels).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos
        feature = self.norm(self.blocks(x))[:, 0]
        if self.proprio_dim:
            if proprio is None:
                raise ValueError('proprio is required by this encoder')
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f'expected proprio_dim={self.proprio_dim}, '
                    f'got {proprio.shape[-1]}')
            feature = torch.cat((feature, proprio.float()), dim=-1)
        return self.projector(feature)


class ProprioEncoder(nn.Module):
    """MLP encoder that maps proprio state → latent, ignoring image input."""

    def __init__(self, proprio_dim: int, latent_dim: int = 192):
        super().__init__()
        if proprio_dim <= 0:
            raise ValueError('ProprioEncoder requires proprio_dim > 0')
        self.proprio_dim = proprio_dim
        self.out_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, latent_dim))

    def forward(self, pixels, proprio=None):
        if proprio is None:
            raise ValueError('ProprioEncoder requires proprio input')
        return self.net(proprio.float())


class StateViTEncoder(nn.Module):
    """Transformer over per-feature state tokens (no image input).

    Each proprio dimension becomes one token so attention learns cross-feature
    interactions. Architecture mirrors PatchViTEncoder for a fair comparison:
    same depth/heads/projector, only the patch embedding is replaced by a
    per-scalar linear embedding.
    """

    def __init__(self, proprio_dim: int, embed_dim: int = 192, depth: int = 4,
                 heads: int = 3, latent_dim: int = 192):
        super().__init__()
        if proprio_dim <= 0:
            raise ValueError('StateViTEncoder requires proprio_dim > 0')
        self.proprio_dim = proprio_dim
        # one token per state dimension
        self.feature_embed = nn.Linear(1, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, proprio_dim + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, heads, int(embed_dim * 4), batch_first=True,
            norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, latent_dim))
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, pixels, proprio=None):
        if proprio is None:
            raise ValueError('StateViTEncoder requires proprio input')
        b = proprio.shape[0]
        # (b, proprio_dim) → (b, proprio_dim, 1) → (b, proprio_dim, embed_dim)
        tokens = self.feature_embed(proprio.float().unsqueeze(-1))
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat((cls, tokens), dim=1) + self.pos
        feature = self.norm(self.blocks(x))[:, 0]
        return self.projector(feature)


class ActionEmbedder(nn.Module):
    """Per-token action embedder used by the original planning model."""

    def __init__(self, action_dim, embed_dim, mlp_scale=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, mlp_scale * embed_dim), nn.SiLU(),
            nn.Linear(mlp_scale * embed_dim, embed_dim))

    def forward(self, action):
        return self.net(action.float())


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Causal Transformer block with AdaLN-zero action conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c):
        s1, k1, g1, s2, k2, g2 = self.adaln(c).chunk(6, dim=-1)
        q = modulate(self.norm1(x), s1, k1)
        b, t, _ = q.shape
        q, k, v = self.to_qkv(q).chunk(3, dim=-1)
        q = q.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
        k = k.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
        v = v.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
        h = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.,
            is_causal=True)
        h = self.to_out(h.transpose(1, 2).reshape(b, t, -1))
        x = x + g1 * h
        x = x + g2 * self.mlp(modulate(self.norm2(x), s2, k2))
        return x


class ARPredictor(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=192, history_size=1,
                 depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=.1):
        super().__init__()
        self.history_size = history_size
        self.input_proj = (nn.Linear(latent_dim, hidden_dim)
                           if latent_dim != hidden_dim else nn.Identity())
        self.cond_proj = (nn.Linear(latent_dim, hidden_dim)
                          if latent_dim != hidden_dim else nn.Identity())
        self.pos = nn.Parameter(torch.randn(1, history_size, latent_dim))
        self.blocks = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = (nn.Linear(hidden_dim, latent_dim)
                       if hidden_dim != latent_dim else nn.Identity())
        self.pred_projector = nn.Sequential(
            nn.Linear(latent_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, latent_dim))

    def forward(self, z, action_embedding):
        x = self.input_proj(z + self.pos[:, :z.shape[1]])
        c = self.cond_proj(action_embedding)
        for block in self.blocks:
            x = block(x, c)
        x = self.output(self.norm(x))
        b, t, d = x.shape
        return self.pred_projector(x.reshape(b * t, d)).reshape(b, t, d)


class InverseModel(nn.Module):
    def __init__(self, latent_dim=192, action_dim=5, hidden_dim=256, seq_len=2):
        super().__init__()
        self.seq_len = int(seq_len)   # number of input latents
        self.action_dim = int(action_dim)
        # Takes seq_len latents, predicts seq_len-1 actions.
        # seq_len=2 → original (z_t, z_{t+1}) → a_t behaviour.
        self.net = nn.Sequential(
            nn.Linear(self.seq_len * latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, (self.seq_len - 1) * action_dim))

    def forward(self, z_seq):
        # z_seq: (b, seq_len, latent_dim)
        b = z_seq.shape[0]
        return self.net(z_seq.flatten(1)).reshape(b, self.seq_len - 1, self.action_dim)


class EndpointInverseModel(nn.Module):
    """Action decoder that sees only the start and end latents (z_t, z_{t+H}).

    Reconstructs the full H-step action sequence from two endpoint
    representations only — no intermediate latents are accessed.
    Architecture mirrors InverseModel: 3-layer MLP, same hidden_dim.
    """
    def __init__(self, latent_dim=192, action_dim=5, hidden_dim=256, horizon=3):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.horizon * action_dim))

    def forward(self, z_start, z_end):
        # z_start, z_end: (b, latent_dim) — endpoints only
        x = torch.cat([z_start, z_end], dim=-1)   # (b, 2*latent_dim)
        b = x.shape[0]
        return self.net(x).reshape(b, self.horizon, self.action_dim)


class StateDecoder(nn.Module):
    """MLP decoder from z_t to normalized physical state. Mirrors InverseModel
    but takes only z_t (not the pair) and outputs state_dim instead of action_dim."""
    def __init__(self, latent_dim=192, state_dim=4, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim))

    def forward(self, z):
        return self.net(z)


class DINOv2Encoder(nn.Module):
    """Frozen pretrained DINOv2 backbone with an optional trainable projector.

    Without projector: CLS token (+ proprio) is used directly — nothing
    can collapse, but no adaptation is possible.
    With projector: a 2-layer MLP maps backbone_out → latent_dim; gradients
    flow through the projector while the backbone stays frozen.
    """
    def __init__(self, variant='dinov2_vits14', proprio_dim=0,
                 use_projector=False, latent_dim=192):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', variant, pretrained=True)
        self.backbone.requires_grad_(False)
        self.proprio_dim = int(proprio_dim)
        backbone_out = self.backbone.embed_dim + self.proprio_dim
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(backbone_out, 2 * latent_dim),
                nn.LayerNorm(2 * latent_dim),
                nn.GELU(),
                nn.Linear(2 * latent_dim, latent_dim))
            self.out_dim = latent_dim
        else:
            self.projector = None
            self.out_dim = backbone_out

    def forward(self, pixels, proprio=None):
        feature = self.backbone.forward_features(pixels)['x_norm_clstoken']
        if self.proprio_dim:
            if proprio is None:
                raise ValueError('proprio required by DINOv2Encoder')
            feature = torch.cat((feature, proprio.float()), dim=-1)
        if self.projector is not None:
            feature = self.projector(feature)
        return feature


class IBOTEncoder(nn.Module):
    """Frozen iBOT-pretrained ViT backbone with an optional trainable projector.

    Loads from an iBOT checkpoint (.pth) which stores weights under the
    'teacher' key with a 'backbone.' prefix, following the official iBOT
    release format.  Uses timm to construct the ViT architecture.
    With use_projector=True a 2-layer MLP maps backbone_out → latent_dim;
    gradients flow through the projector while the backbone stays frozen.
    """
    def __init__(self, checkpoint_path, arch='vit_small_patch16_224',
                 proprio_dim=0, use_projector=False, latent_dim=192):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError('timm is required for IBOTEncoder: pip install timm')
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # iBOT stores weights under 'teacher'; strip the 'backbone.' prefix.
        raw = ckpt.get('teacher', ckpt.get('student', ckpt))
        sd = {k[len('backbone.'):]: v
              for k, v in raw.items() if k.startswith('backbone.')}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        if missing:
            print(f'[IBOTEncoder] missing keys ({len(missing)}): {missing[:5]} …')
        self.backbone.requires_grad_(False)
        self.proprio_dim = int(proprio_dim)
        backbone_out = self.backbone.embed_dim + self.proprio_dim
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(backbone_out, 2 * latent_dim),
                nn.LayerNorm(2 * latent_dim),
                nn.GELU(),
                nn.Linear(2 * latent_dim, latent_dim))
            self.out_dim = latent_dim
        else:
            self.projector = None
            self.out_dim = backbone_out

    def forward(self, pixels, proprio=None):
        feat = self.backbone.forward_features(pixels)
        # timm ViT returns (b, seq_len, d); CLS token is at index 0.
        if feat.ndim == 3:
            feat = feat[:, 0]
        if self.proprio_dim:
            if proprio is None:
                raise ValueError('proprio required by IBOTEncoder')
            feat = torch.cat((feat, proprio.float()), dim=-1)
        if self.projector is not None:
            feat = self.projector(feat)
        return feat


class SensorimotorWorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg.get('latent_dim', 192))
        action_dim = int(cfg.get('action_context_dim', 5))
        self.action_context_dim = action_dim
        self.history_size = int(cfg.get('history_size', 1))
        self.use_frame_diff = bool(cfg.get('use_frame_diff', False))
        self.frame_stack = int(cfg.get('frame_stack', 1))
        self.use_proprio = bool(cfg.get('use_proprio', False))
        self.proprio_dim = int(cfg.get('proprio_dim', 4)) if self.use_proprio else 0
        # Indices into the full 4-D state to use as proprio input.
        # None means use all dimensions (default, backward-compatible).
        raw_idx = cfg.get('proprio_indices', None)
        self.proprio_indices = list(map(int, raw_idx)) if raw_idx is not None else None
        encoder_type = str(cfg.get('encoder_type', 'vit'))
        if encoder_type == 'state_vit':
            self.encoder = StateViTEncoder(
                proprio_dim=self.proprio_dim,
                embed_dim=int(cfg.get('state_vit_embed_dim', 192)),
                depth=int(cfg.get('state_vit_depth', 4)),
                heads=int(cfg.get('state_vit_heads', 3)),
                latent_dim=d)
        elif encoder_type == 'proprio':
            self.encoder = ProprioEncoder(
                proprio_dim=self.proprio_dim,
                latent_dim=d)
        elif encoder_type == 'dinov2':
            self.encoder = DINOv2Encoder(
                variant=str(cfg.get('dinov2_variant', 'dinov2_vits14')),
                proprio_dim=self.proprio_dim,
                use_projector=bool(cfg.get('use_projector', False)),
                latent_dim=d)
            d = self.encoder.out_dim
        elif encoder_type == 'ibot':
            self.encoder = IBOTEncoder(
                checkpoint_path=str(cfg['ibot_ckpt']),
                arch=str(cfg.get('ibot_arch', 'vit_small_patch16_224')),
                proprio_dim=self.proprio_dim,
                use_projector=bool(cfg.get('use_projector', False)),
                latent_dim=d)
            d = self.encoder.out_dim
        else:
            self.encoder = PatchViTEncoder(
                image_size=int(cfg.get('image_size', 128)),
                patch_size=int(cfg.get('patch_size', 8)),
                in_channels=9 if self.use_frame_diff else 3 * self.frame_stack,
                proprio_dim=self.proprio_dim,
                embed_dim=int(cfg.get('vit_embed_dim', 192)),
                depth=int(cfg.get('vit_depth', 4)),
                heads=int(cfg.get('vit_heads', 3)),
                latent_dim=d)
        self.action_encoder = ActionEmbedder(action_dim, d)
        self.predictor = ARPredictor(
            latent_dim=d, hidden_dim=int(cfg.get('predictor_hidden_dim', 192)),
            history_size=self.history_size,
            depth=int(cfg.get('predictor_depth', 6)),
            heads=int(cfg.get('predictor_heads', 16)),
            dim_head=int(cfg.get('predictor_dim_head', 64)),
            mlp_dim=int(cfg.get('predictor_mlp_dim', 2048)),
            dropout=float(cfg.get('predictor_dropout', .1)))
        self.use_inverse_model = bool(cfg.get('use_inverse_model', True))
        self.inverse_model = (InverseModel(
            d, action_dim,
            int(cfg.get('inverse_hidden_dim', 256)),
            seq_len=int(cfg.get('inverse_seq_len', 2)))
            if self.use_inverse_model else None)
        # Optional separate single-step inverse head (seq_len always = 2).
        self.use_inverse_model_1step = bool(cfg.get('use_inverse_model_1step', False))
        self.inverse_model_1step = (InverseModel(
            d, action_dim,
            int(cfg.get('inverse_1step_hidden_dim', 256)),
            seq_len=2)
            if self.use_inverse_model_1step else None)
        self.use_endpoint_inverse = bool(cfg.get('use_endpoint_inverse', False))
        self.endpoint_inverse_model = (EndpointInverseModel(
            d, action_dim,
            int(cfg.get('inverse_hidden_dim', 256)),
            horizon=int(cfg.get('endpoint_inverse_horizon', 3)))
            if self.use_endpoint_inverse else None)
        self.use_state_decoder = bool(cfg.get('use_state_decoder', False))
        self.state_decoder = (StateDecoder(
            d, int(cfg.get('state_dim', 4)),
            int(cfg.get('state_decoder_hidden_dim', 256)))
            if self.use_state_decoder else None)
        # When True: action context is the history [a_{t-ctx+1},...,a_t].
        # When False (default): context is the current action repeated ctx times.
        self.use_action_history = bool(cfg.get('use_action_history', False))

    def encode(self, obs, prev_obs=None, proprio=None):
        if self.use_frame_diff:
            if prev_obs is None:
                prev_obs = obs
            obs = torch.cat((prev_obs, obs, obs - prev_obs), dim=1)
        # frame_stack > 1: obs already has shape (b, 3*frame_stack, H, W) from dataset
        return self.encoder(obs, proprio)

    def expand_action(self, action):
        # action: (batch, d) → (batch, action_context_dim).
        # d=1 (CartPole): backward-compatible behaviour.
        # d>1 (e.g. PointMaze 2-D action): tiles the d-dim vector to fill ctx.
        ctx = self.action_context_dim
        d = action.shape[-1]
        if self.use_action_history:
            pad = action.new_zeros(action.shape[0], ctx - d)
            return torch.cat([pad, action], dim=-1)
        reps = math.ceil(ctx / d)
        return action.repeat(1, reps)[:, :ctx]

    def predict(self, z_history, action_history):
        # action_history: (b, t, action_context_dim) — full context per step.
        b, t, ctx = action_history.shape
        action = self.action_encoder(action_history.reshape(b * t, ctx))
        return self.predictor(z_history, action.reshape(b, t, -1))
