"""DSMIL bag aggregator (Li et al., CVPR 2021) — dual-stream MIL on frame features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSMILAggregator(nn.Module):
    """
    Stream A: instance classifier → critical (max) instance.
    Stream B: attention over instances related to the critical one.
    Returns frame weights for pooling (+ optional fused bag feature).
    """

    def __init__(self, in_dim: int = 256, num_classes: int = 2, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.num_classes = int(num_classes)
        self.i_classifier = nn.Linear(in_dim, num_classes)
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, attn_dim),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        feat: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        fuse: str = "mean",
    ):
        """
        feat: [B, N, D]
        returns:
          frame_w: [B, N]
          bag_feat: [B, D]
          crit_idx: [B]
        """
        b, n, d = feat.shape
        feat = self.dropout(feat)
        logits = self.i_classifier(feat)  # [B, N, K]
        # critical instance = max positive logit (class 1) among valid frames
        score = logits[..., 1]
        if frame_mask is not None:
            m = frame_mask.to(dtype=score.dtype)
            score = score.masked_fill(m < 0.5, -1e9)
        crit_idx = score.argmax(dim=1)  # [B]
        crit_feat = feat[torch.arange(b, device=feat.device), crit_idx]  # [B, D]

        q_crit = self.q(crit_feat).unsqueeze(1)  # [B, 1, A]
        v_all = self.v(feat)  # [B, N, A]
        # similarity attention (DSMIL-style)
        att = torch.matmul(v_all, q_crit.transpose(-1, -2)).squeeze(-1) / (v_all.shape[-1] ** 0.5)
        if frame_mask is not None:
            att = att.masked_fill(frame_mask < 0.5, -1e9)
        frame_w = F.softmax(att, dim=1)
        if frame_mask is not None:
            frame_w = frame_w * frame_mask.to(dtype=frame_w.dtype)
            frame_w = frame_w / frame_w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        bag_attn = torch.sum(frame_w.unsqueeze(-1) * feat, dim=1)
        bag_max = crit_feat
        if fuse == "max":
            bag_feat = bag_max
        elif fuse == "attn":
            bag_feat = bag_attn
        else:
            bag_feat = 0.5 * (bag_max + bag_attn)
        return frame_w, bag_feat, crit_idx
