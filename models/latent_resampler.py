import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_anchor_grid(num_latents):
    """Evenly spaced 2D anchors in [0,1]^2, one per latent slot."""
    side = math.ceil(math.sqrt(num_latents))
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, side), torch.linspace(0, 1, side), indexing='ij'
    )
    grid = torch.stack([xs.flatten(), ys.flatten()], dim=-1)  # [side*side, 2]
    return grid[:num_latents].contiguous()


class SpatiallyBiasedCrossAttention(nn.Module):
    """
    Spatially-biased multi-head cross-attention.
        gamma_init = -6.0 -> gamma 0.0025, sigma 14.2  (flat: no prior at all)
        gamma_init =  8.0 -> gamma 8.0,    sigma 0.25  (~the 0.2 anchor-grid pitch)
    """

    def __init__(self, d_model, vision_width, num_heads, num_patches_side,
                 gamma_init=-6.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(vision_width, d_model)
        self.v_proj = nn.Linear(vision_width, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        side = num_patches_side
        ys, xs = torch.meshgrid(
            torch.arange(side), torch.arange(side), indexing='ij'
        )
        patch_coords = torch.stack([xs.flatten(), ys.flatten()], dim=-1).float()
        patch_coords = patch_coords / max(side - 1, 1)
        cls_coord = torch.full((1, 2), 0.5)
        patch_coords = torch.cat([cls_coord, patch_coords], dim=0)  # [N_vis, 2]
        self.register_buffer('patch_coords', patch_coords, persistent=False)

        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, latents, anchors, visual_features):
        B, N_lat, _ = latents.shape
        N_vis = visual_features.shape[1]

        q = self.q_proj(latents).view(B, N_lat, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(visual_features).view(B, N_vis, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(visual_features).view(B, N_vis, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, N_lat, N_vis]

        patch_coords = self.patch_coords.to(anchors.dtype).unsqueeze(0).expand(B, -1, -1)
        dist2 = torch.cdist(anchors, patch_coords) ** 2  # [B, N_lat, N_vis]
        dist2 = dist2.clone()
        dist2[:, :, 0] = 0.0  # neutralize bias for the CLS (global) token
        gamma = F.softplus(self.log_gamma)
        attn_scores = attn_scores + (-gamma * dist2).unsqueeze(1)

        attn_weights = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn_weights, v)  # [B, h, N_lat, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, N_lat, -1)
        return self.out_proj(out)


class LatentCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, vision_width, num_heads, num_patches_side, ff_mult=4,
                 gamma_init=-6.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(vision_width)
        self.cross_attn = SpatiallyBiasedCrossAttention(d_model, vision_width, num_heads,
                                                        num_patches_side, gamma_init=gamma_init)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, latents, anchors, visual_features):
        kv = self.norm_kv(visual_features)
        attn_out = self.cross_attn(self.norm_q(latents), anchors, kv)
        latents = latents + attn_out
        latents = latents + self.ff(self.norm_ff(latents))
        return latents


class LatentResampler(nn.Module):
    """
    Latent resampler: compress N_vis visual tokens → N_lat latent tokens via
    stacked spatially-anchored cross-attention layers, then feed into the
    caption decoder.
    """

    def __init__(self, num_latent_tokens, d_model, vision_width, num_heads,
                 num_layers=2, num_patches_side=24, gamma_init=-6.0):
        super().__init__()
        # l_q: latent query
        self.latent_query = nn.Parameter(torch.empty(1, num_latent_tokens, d_model))
        nn.init.trunc_normal_(self.latent_query, std=0.02)

        # a_t: anchor 2D tokens
        self.anchors = nn.Parameter(_init_anchor_grid(num_latent_tokens))

        self.layers = nn.ModuleList([
            LatentCrossAttentionLayer(d_model, vision_width, num_heads, num_patches_side,
                                      gamma_init=gamma_init)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_features):
        """
        visual_features : [B, N_vis, vision_width]   t_e, from the image encoder
        returns          : [B, N_lat, d_model]        q_z, to the caption decoder
        """
        B = visual_features.size(0)
        latents = self.latent_query.expand(B, -1, -1)
        anchors = self.anchors.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, anchors, visual_features)
        return self.norm(latents)
