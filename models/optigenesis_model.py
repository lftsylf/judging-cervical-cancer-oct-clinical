import torch
import torch.nn as nn
import timm
from .uncertainty import UncertaintyHead

class OptiGenesis(nn.Module):
    """
    OptiGenesis Model: Lancet Edition
    架构: Swin Transformer (Vision) + MLP (Clinical) -> Fusion -> EDL Head
    """
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=2, use_clinical=True):
        super().__init__()
        self.use_clinical = use_clinical
        
        # 1. 视觉基座 (Swin Transformer)
        print(f"🔍 Loading Vision Backbone: {model_name}")
        # num_classes=0 表示移除原本的分类头，只提取特征
        self.vision_backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.vision_dim = self.vision_backbone.num_features # Swin-Tiny通常是768
        
        # 2. 临床数据编码器 (MLP)
        if self.use_clinical:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(3, 32), # 输入: Age, HPV, TCT
                nn.LayerNorm(32),  # 使用LayerNorm替代BatchNorm以支持batch_size=1
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.LayerNorm(64),
                nn.ReLU()
            )
            fusion_input_dim = self.vision_dim + 64
        else:
            fusion_input_dim = self.vision_dim
            
        # 3. 多模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2) # 防止过拟合
        )
        
        # 4. 不确定性输出头
        self.uncertainty_head = UncertaintyHead(in_features=256, num_classes=num_classes)
        # 辅助头：用于单模态辅助监督（低成本多模态改进）
        self.aux_vision_head = nn.Linear(self.vision_dim, num_classes)
        if self.use_clinical:
            self.aux_clinical_head = nn.Linear(64, num_classes)
        else:
            self.aux_clinical_head = None

    def forward(self, img, clinical, return_aux=False, return_coral_feat=False):
        # A. 提取视觉特征 (处理多图像)
        # img shape: [B, N_images, C, H, W]
        B, N_images, C, H, W = img.shape
        img_flat = img.view(B * N_images, C, H, W)  # [B*N, C, H, W]
        
        # 提取所有图像的特征
        v_feat_flat = self.vision_backbone(img_flat)  # [B*N, 768]
        v_feat = v_feat_flat.view(B, N_images, -1)  # [B, N, 768]
        
        # 多图像特征融合：平均池化
        v_feat = torch.mean(v_feat, dim=1)  # [B, 768]
        
        # B. 提取/融合临床特征
        if self.use_clinical:
            c_feat = self.clinical_mlp(clinical) # [B, 64]
            feat = torch.cat([v_feat, c_feat], dim=1) # [B, 832]
        else:
            c_feat = None
            feat = v_feat
            
        # C. 特征融合（与 EDL 头输入一致，适合 CORAL 对齐）
        feat_fused = self.fusion_layer(feat) # [B, 256]
        
        # D. 输出 Dirichlet 参数 Alpha
        alpha = self.uncertainty_head(feat_fused)
        if return_aux:
            aux_logits_vision = self.aux_vision_head(v_feat)
            aux_logits_clinical = (
                self.aux_clinical_head(c_feat) if self.use_clinical else None
            )
            if return_coral_feat:
                return alpha, aux_logits_vision, aux_logits_clinical, feat_fused
            return alpha, aux_logits_vision, aux_logits_clinical
        if return_coral_feat:
            return alpha, feat_fused
        return alpha
