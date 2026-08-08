"""
UBIX: Uncertainty-Based Instance eXclusion (de Vente et al., MedIA 2024).

Inference-time reweighting before MIL pooling. Uncertainty via MC Dropout
(not EDL-u — keep fair vs Ours UWA).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _enable_mc_dropout(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def estimate_frame_uncertainty_mc(
    model: nn.Module,
    v_feat: torch.Tensor,
    clinical: torch.Tensor,
    frame_mask: torch.Tensor | None,
    mc_t: int = 16,
) -> torch.Tensor:
    """
    MC Dropout predictive entropy on frame-level class probs.
    Returns u in [0, 1]-ish: H / log(K), shape [B, N].
    """
    was_training = model.training
    model.eval()
    _enable_mc_dropout(model.fusion_layer)
    if getattr(model, "ubix_dropout", None) is not None:
        model.ubix_dropout.train()

    probs_stack = []
    t = max(1, int(mc_t))
    for _ in range(t):
        alpha_frame, _, _, _ = model._frame_alphas_and_features(
            v_feat, clinical, frame_mask=frame_mask
        )
        if getattr(model, "ubix_dropout", None) is not None:
            # re-draw after dropout on fused path: redo head on dropped feats
            b, n, _ = v_feat.shape
            # use fused from a fresh pass with explicit dropout
            alpha_frame, feat_fused, _, _ = model._frame_alphas_and_features(
                v_feat, clinical, frame_mask=frame_mask
            )
            feat_d = model.ubix_dropout(feat_fused)
            alpha_frame = model.uncertainty_head(feat_d.reshape(b * n, -1)).view(b, n, -1)
        s = alpha_frame.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        p = alpha_frame / s
        probs_stack.append(p)

    p_mean = torch.stack(probs_stack, dim=0).mean(dim=0).clamp_min(1e-8)  # [B,N,K]
    ent = -(p_mean * p_mean.log()).sum(dim=-1)  # [B,N]
    u = ent / float(torch.log(torch.tensor(float(p_mean.shape[-1]), device=ent.device)))
    if frame_mask is not None:
        u = u * frame_mask.to(dtype=u.dtype)

    if was_training:
        model.train()
    else:
        model.eval()
    return u


def ubix_weights(
    u: torch.Tensor,
    frame_mask: torch.Tensor | None,
    mode: str = "soft",
    thresh: float | None = None,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Map instance uncertainty → pooling weights (higher u → lower weight)."""
    mode = (mode or "soft").lower()
    b, n = u.shape
    if mode == "hard":
        thr = 0.5 if thresh is None or thresh == "" else float(thresh)
        keep = (u <= thr).to(dtype=u.dtype)
        if frame_mask is not None:
            keep = keep * frame_mask.to(dtype=u.dtype)
        # if all excluded, fall back to equal over valid
        denom = keep.sum(dim=1, keepdim=True)
        fallback = frame_mask if frame_mask is not None else torch.ones_like(u)
        need_fb = (denom < 0.5).float()
        keep = keep + need_fb * fallback
        w = keep / keep.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return w

    # soft: softmax((1-u)/τ)
    tau = max(float(temperature), 1e-6)
    score = (1.0 - u).clamp(min=0.0)
    if frame_mask is not None:
        score = score.masked_fill(frame_mask < 0.5, -1e9)
    w = F.softmax(score / tau, dim=1)
    if frame_mask is not None:
        w = w * frame_mask.to(dtype=w.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return w


@torch.no_grad()
def ubix_reaggregate(
    model: nn.Module,
    v_feat: torch.Tensor,
    clinical: torch.Tensor,
    frame_mask: torch.Tensor | None,
    alpha_frame: torch.Tensor,
    feat_fused: torch.Tensor,
    *,
    mode: str = "soft",
    mc_t: int = 16,
    thresh: float | None = None,
    temperature: float = 0.5,
):
    """Estimate u via MC Dropout, reweight, pool frame α / features."""
    u = estimate_frame_uncertainty_mc(
        model, v_feat, clinical, frame_mask, mc_t=mc_t
    )
    w = ubix_weights(u, frame_mask, mode=mode, thresh=thresh, temperature=temperature)
    ww = w.unsqueeze(-1)
    alpha = torch.sum(ww * alpha_frame, dim=1)
    feat_patient = torch.sum(ww * feat_fused, dim=1)
    return alpha, feat_patient, u, w
