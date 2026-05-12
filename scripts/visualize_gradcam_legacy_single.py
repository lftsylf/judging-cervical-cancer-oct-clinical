#!/usr/bin/env python3
"""
Legacy single-image EigenCAM (v1-style) for compatibility comparison.
This script intentionally keeps the earlier conservative behavior:
- single target layer: Stage4_LastBlock (norm1)
- no percentile contrast enhancement
- fixed target class = 1
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.optigenesis_model import OptiGenesis


def read_image_rgb(path: str) -> np.ndarray:
    # PIL first
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        pass
    # OpenCV fallback
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo + 1e-8) * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        # likely BHWC
        if tensor.shape[-1] in (96, 192, 384, 768, 1024):
            return tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.contiguous()
    if tensor.ndim == 3:
        b, n, c = tensor.shape
        h = int(np.sqrt(n))
        w = h if h * h == n else max(1, n // max(1, h))
        return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected tensor shape: {tuple(tensor.shape)}")


def main() -> None:
    checkpoint = "outputs/消融实验/outputs_wma_ema_aux/xiangya/seed_2024/checkpoints/best_model.pth"
    image_path = "case_study_4_panels/2_Huaxi_Easy_TN/M0008_2021_P0000314_circle_2.4x3.0_C8_S9.tiff"
    out_path = "figures/paper/gradcam_4cases/M0008_2021_P0000314_circle_2.4x3.0_C8_S9_gradcam_legacy_v1.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model = OptiGenesis(model_name="swin_tiny_patch4_window7_224", use_clinical=True, num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()

    # v1 behavior: fixed clinical prior and class-1 target
    clinical = torch.tensor([[40.0, 1.0, 1.0]], dtype=torch.float32, device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m, c):
            super().__init__()
            self.m = m
            self.c = c

        def forward(self, x):
            return self.m(x.unsqueeze(1), self.c.expand(x.shape[0], -1))

    wrapped = Wrapper(model, clinical).to(device).eval()
    target_layer = model.vision_backbone.layers[-1].blocks[-1].norm1
    cam = EigenCAM(model=wrapped, target_layers=[target_layer], reshape_transform=reshape_transform)

    rgb = read_image_rgb(image_path)
    img = cv2.resize(rgb, (224, 224))
    img_f = np.float32(img) / 255.0

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tfm(Image.fromarray(rgb)).unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(1)])[0, :]
    overlay = show_cam_on_image(img_f, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_f)
    axes[0].set_title("Original OCT Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("EigenCAM Heatmap (Legacy v1)", fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
