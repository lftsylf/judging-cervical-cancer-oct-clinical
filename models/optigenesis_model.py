import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import timm
from .uncertainty import UncertaintyHead


class OptiGenesis(nn.Module):
    """
    OptiGenesis（Lancet / MUSE 实验线）

    视觉骨干逐帧编码后，按 FRAME_AGG_MODE 聚合到患者级，再（或同时）做 EDL：

    - ``mean``：等权均值池化特征 → 患者级 EDL（v3 / B1 对照）
    - ``equal``：帧级 EDL → 等权平均 α（消融：有帧不确定、无加权）
    - ``uncertainty_weighted``：帧级 EDL → 按 (1−u) 软加权聚合 α（v4 主方法）

    帧不确定度（EDL）：u = K / S，S = Σα；置信度 (1−u) 越大，聚合权重越高。
    """

    AGG_MODES = ("mean", "equal", "uncertainty_weighted")
    WEIGHT_SIGNALS = ("edl_u", "edl_u_amp", "maxprob", "negent", "topk_p", "max_p_pool")

    def __init__(
        self,
        model_name="resnet50",
        num_classes=2,
        use_clinical=True,
        frame_agg_mode="uncertainty_weighted",
        agg_temperature=0.5,
        review_top_k=3,
        weight_signal="edl_u",
        u_score_base=0.5,
        u_score_scale=10.0,
        frame_encode_chunk=16,
        frame_topk_k=5,
    ):
        super().__init__()
        self.use_clinical = use_clinical
        self.backbone_name = model_name
        self.num_classes = int(num_classes)
        self.frame_agg_mode = str(frame_agg_mode)
        if self.frame_agg_mode not in self.AGG_MODES:
            raise ValueError(
                f"frame_agg_mode 必须是 {self.AGG_MODES} 之一，收到: {frame_agg_mode}"
            )
        self.agg_temperature = float(agg_temperature)
        self.review_top_k = int(review_top_k)
        self.weight_signal = str(weight_signal).strip().lower()
        if self.weight_signal not in self.WEIGHT_SIGNALS:
            raise ValueError(
                f"weight_signal 必须是 {self.WEIGHT_SIGNALS} 之一，收到: {weight_signal}"
            )
        self.u_score_base = float(u_score_base)
        self.u_score_scale = float(u_score_scale)
        self.frame_encode_chunk = int(frame_encode_chunk or 0)
        self.frame_topk_k = max(1, int(frame_topk_k))

        # 1. 视觉基座（timm；num_classes=0 去掉分类头，前向得到全局池化后的特征向量）
        print(f"🔍 正在加载视觉 backbone: {model_name}")
        print(
            f"   帧聚合模式 frame_agg_mode={self.frame_agg_mode} | "
            f"weight_signal={self.weight_signal} | τ={self.agg_temperature}"
            + (
                f" | u_base={self.u_score_base} scale={self.u_score_scale}"
                if self.weight_signal == "edl_u_amp"
                else ""
            )
            + (f" | topk_k={self.frame_topk_k}" if self.weight_signal == "topk_p" else "")
        )
        self.vision_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vision_dim = self.vision_backbone.num_features

        # 2. 临床数据编码器 (MLP)
        if self.use_clinical:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(3, 32),  # 输入: Age, HPV, TCT
                nn.LayerNorm(32),  # 使用 LayerNorm 替代 BatchNorm 以支持 batch_size=1
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
            )
            fusion_input_dim = self.vision_dim + 64
        else:
            fusion_input_dim = self.vision_dim

        # 3. 多模态融合层（mean 模式：患者级一次；帧级模式：逐帧共用）
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 4. EDL 头（mean：患者级；equal / uncertainty_weighted：帧级共用同一头）
        self.uncertainty_head = UncertaintyHead(in_features=256, num_classes=num_classes)

        # 辅助头：用于单模态辅助监督（低成本多模态改进；仍基于患者级视觉/临床特征）
        self.aux_vision_head = nn.Linear(self.vision_dim, num_classes)
        if self.use_clinical:
            self.aux_clinical_head = nn.Linear(64, num_classes)
        else:
            self.aux_clinical_head = None

    @staticmethod
    def _dirichlet_uncertainty(alpha: torch.Tensor) -> torch.Tensor:
        """EDL 不确定度 u = K / Σα。alpha: [..., K] → u: [...]"""
        k = alpha.shape[-1]
        s = torch.sum(alpha, dim=-1).clamp_min(1e-8)
        return float(k) / s

    def _frame_alphas_and_features(
        self, v_feat: torch.Tensor, clinical: torch.Tensor, frame_mask: torch.Tensor | None = None
    ):
        """
        逐帧：特征 →（可选临床拼接）→ fusion → EDL α。

        v_feat: [B, N, Dv]
        frame_mask: [B, N]，1=有效帧（展开多页后 batch pad 用）
        返回:
            alpha_frame: [B, N, K]
            feat_fused:  [B, N, 256]
            c_feat:      [B, 64] 或 None
            v_feat_patient: [B, Dv]  （mask 加权均值，供 Aux 使用）
        """
        b, n, _ = v_feat.shape
        if frame_mask is None:
            v_feat_patient = torch.mean(v_feat, dim=1)
        else:
            m = frame_mask.to(dtype=v_feat.dtype).unsqueeze(-1)  # [B,N,1]
            denom = m.sum(dim=1).clamp_min(1.0)
            v_feat_patient = (v_feat * m).sum(dim=1) / denom

        if self.use_clinical:
            c_feat = self.clinical_mlp(clinical)  # [B, 64]
            c_exp = c_feat.unsqueeze(1).expand(-1, n, -1)  # [B, N, 64]
            feat = torch.cat([v_feat, c_exp], dim=-1)  # [B, N, Dv+64]
        else:
            c_feat = None
            feat = v_feat

        feat_flat = feat.reshape(b * n, -1)
        feat_fused = self.fusion_layer(feat_flat).view(b, n, -1)  # [B, N, 256]
        alpha_frame = self.uncertainty_head(
            feat_fused.reshape(b * n, -1)
        ).view(b, n, self.num_classes)
        return alpha_frame, feat_fused, c_feat, v_feat_patient

    def _aggregate_frame_alphas(
        self,
        alpha_frame: torch.Tensor,
        feat_fused: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ):
        """
        由帧级 α / 特征得到患者级 α 与融合特征。

        返回:
            alpha: [B, K]
            feat_patient: [B, 256]
            frame_u: [B, N]
            frame_w: [B, N]
        """
        frame_u = self._dirichlet_uncertainty(alpha_frame)  # [B, N]
        b, n, _ = alpha_frame.shape
        tau = max(float(self.agg_temperature), 1e-6)
        mask = None
        if frame_mask is not None:
            mask = frame_mask.to(dtype=alpha_frame.dtype)
            # padding 帧不确定度置 0，避免 review/导出误导
            frame_u = frame_u * mask

        if self.frame_agg_mode == "equal":
            if mask is None:
                frame_w = torch.full(
                    (b, n), 1.0 / float(n), device=alpha_frame.device, dtype=alpha_frame.dtype
                )
            else:
                frame_w = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        elif self.weight_signal in ("topk_p", "max_p_pool"):
            # 按帧阳性概率硬选 top-k / 最大帧，等权（训练期梯度只流经选中帧）
            s = torch.sum(alpha_frame, dim=-1, keepdim=True).clamp_min(1e-8)
            p_pos = (alpha_frame / s)[..., 1]  # [B, N]
            if mask is not None:
                p_pos = p_pos.masked_fill(mask < 0.5, -1e9)
            if self.weight_signal == "max_p_pool":
                k = 1
            else:
                k = int(min(self.frame_topk_k, n))
            idx = torch.topk(p_pos, k=k, dim=1, largest=True).indices  # [B, k]
            frame_w = torch.zeros(b, n, device=alpha_frame.device, dtype=alpha_frame.dtype)
            # 每行等权 1/k_i（k 固定；若有效帧 <k，topk 仍返回 k 个但含 -1e9 帧——用 mask 再清）
            ones = torch.full((b, k), 1.0 / float(k), device=alpha_frame.device, dtype=alpha_frame.dtype)
            frame_w.scatter_(1, idx, ones)
            if mask is not None:
                frame_w = frame_w * mask
                frame_w = frame_w / frame_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            # uncertainty_weighted：按 weight_signal 打分再 softmax(/τ)
            if self.weight_signal == "maxprob":
                s = torch.sum(alpha_frame, dim=-1, keepdim=True).clamp_min(1e-8)
                p = alpha_frame / s
                score = p.max(dim=-1).values  # [B, N]
            elif self.weight_signal == "negent":
                s = torch.sum(alpha_frame, dim=-1, keepdim=True).clamp_min(1e-8)
                p = (alpha_frame / s).clamp_min(1e-8)
                ent = -(p * p.log()).sum(dim=-1)  # [B, N]
                score = -ent
                if mask is None:
                    score = score - score.mean(dim=1, keepdim=True)
                else:
                    # 仅在有效帧上减均值
                    msum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    mean = (score * mask).sum(dim=1, keepdim=True) / msum
                    score = score - mean
            elif self.weight_signal == "edl_u_amp":
                # score = (u_base − u) · scale：把挤在窄区间的 u 差放大后再 /τ
                score = (float(self.u_score_base) - frame_u) * float(self.u_score_scale)
            else:
                # edl_u（默认）：置信度 (1−u)
                score = (1.0 - frame_u).clamp(min=0.0)
            if mask is not None:
                score = score.masked_fill(mask < 0.5, -1e9)
            frame_w = F.softmax(score / tau, dim=1)  # [B, N]
            if mask is not None:
                frame_w = frame_w * mask
                frame_w = frame_w / frame_w.sum(dim=1, keepdim=True).clamp_min(1e-8)

        w = frame_w.unsqueeze(-1)  # [B, N, 1]
        alpha = torch.sum(w * alpha_frame, dim=1)  # [B, K]
        feat_patient = torch.sum(w * feat_fused, dim=1)  # [B, 256]
        return alpha, feat_patient, frame_u, frame_w

    def _review_indices(
        self, frame_u: torch.Tensor, frame_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """每例患者不确定性最高的 top-k 帧下标，形状 [B, top_k]（仅有效帧）。"""
        u = frame_u
        if frame_mask is not None:
            u = u.masked_fill(frame_mask < 0.5, -1.0)
        k = min(self.review_top_k, frame_u.shape[1])
        return torch.topk(u, k=k, dim=1, largest=True).indices

    def _encode_vision_frames(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, N, C, H, W] → v_feat [B, N, Dv]
        N 很大（全时序 60/120）时按 chunk 前向；训练期用 checkpoint 降低显存。
        """
        b, n_images, c, h, w = img.shape
        chunk = int(self.frame_encode_chunk or 0)
        if chunk <= 0 or n_images <= chunk:
            flat = img.view(b * n_images, c, h, w)
            return self.vision_backbone(flat).view(b, n_images, -1)

        feats = []
        for start in range(0, n_images, chunk):
            end = min(start + chunk, n_images)
            flat = img[:, start:end].reshape(-1, c, h, w)
            if self.training:
                # checkpoint：反传时重算该块，避免同时保留全部帧的中间激活
                out = checkpoint(self.vision_backbone, flat, use_reentrant=False)
            else:
                out = self.vision_backbone(flat)
            feats.append(out.view(b, end - start, -1))
        return torch.cat(feats, dim=1)

    def forward(
        self,
        img,
        clinical,
        return_aux=False,
        return_coral_feat=False,
        return_frame_details=False,
        frame_mask=None,
    ):
        # A. 逐帧视觉特征
        # img: [B, N, C, H, W]
        v_feat = self._encode_vision_frames(img)  # [B, N, Dv]
        b, n_images, _ = v_feat.shape

        if frame_mask is not None:
            frame_mask = frame_mask.to(device=img.device, dtype=v_feat.dtype)

        frame_details = None

        if self.frame_agg_mode == "mean":
            # —— v3 / B1：先等权池化，再患者级 EDL ——
            if frame_mask is None:
                v_feat_patient = torch.mean(v_feat, dim=1)
            else:
                m = frame_mask.unsqueeze(-1)
                v_feat_patient = (v_feat * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            if self.use_clinical:
                c_feat = self.clinical_mlp(clinical)
                feat = torch.cat([v_feat_patient, c_feat], dim=1)
            else:
                c_feat = None
                feat = v_feat_patient
            feat_fused = self.fusion_layer(feat)  # [B, 256]
            alpha = self.uncertainty_head(feat_fused)
            if return_frame_details:
                # mean 模式无真正帧级 α；用「等权、u 占位」方便下游接口统一
                frame_u = torch.zeros(b, n_images, device=img.device, dtype=alpha.dtype)
                if frame_mask is None:
                    frame_w = torch.full(
                        (b, n_images),
                        1.0 / float(n_images),
                        device=img.device,
                        dtype=alpha.dtype,
                    )
                else:
                    frame_w = frame_mask / frame_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                frame_details = {
                    "frame_alpha": None,
                    "frame_uncertainty": frame_u,
                    "frame_weights": frame_w,
                    "review_frame_indices": self._review_indices(frame_u, frame_mask),
                    "agg_mode": self.frame_agg_mode,
                    "frame_mask": frame_mask,
                }
        else:
            # —— equal / uncertainty_weighted：帧级 EDL 再聚合 ——
            alpha_frame, feat_fused_frames, c_feat, v_feat_patient = self._frame_alphas_and_features(
                v_feat, clinical, frame_mask=frame_mask
            )
            alpha, feat_fused, frame_u, frame_w = self._aggregate_frame_alphas(
                alpha_frame, feat_fused_frames, frame_mask=frame_mask
            )
            if return_frame_details:
                frame_details = {
                    "frame_alpha": alpha_frame,
                    "frame_uncertainty": frame_u,
                    "frame_weights": frame_w,
                    "review_frame_indices": self._review_indices(frame_u, frame_mask),
                    "agg_mode": self.frame_agg_mode,
                    "frame_mask": frame_mask,
                }

        # 组装返回值（保持旧调用兼容：默认只返回患者级 alpha）
        if return_aux:
            aux_logits_vision = self.aux_vision_head(v_feat_patient)
            aux_logits_clinical = (
                self.aux_clinical_head(c_feat) if self.use_clinical else None
            )
            if return_coral_feat and return_frame_details:
                return alpha, aux_logits_vision, aux_logits_clinical, feat_fused, frame_details
            if return_coral_feat:
                return alpha, aux_logits_vision, aux_logits_clinical, feat_fused
            if return_frame_details:
                return alpha, aux_logits_vision, aux_logits_clinical, frame_details
            return alpha, aux_logits_vision, aux_logits_clinical

        if return_coral_feat and return_frame_details:
            return alpha, feat_fused, frame_details
        if return_coral_feat:
            return alpha, feat_fused
        if return_frame_details:
            return alpha, frame_details
        return alpha
