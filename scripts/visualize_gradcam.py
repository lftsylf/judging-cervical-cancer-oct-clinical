#!/usr/bin/env python3
"""
EigenCAM multi-layer sweep for OptiGenesis (Swin-Tiny backbone).

This script generates one heatmap per candidate layer for each input image:
  [patient_id]_gradcam_[layer_name].png
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.optigenesis_model import OptiGenesis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-layer EigenCAM heatmaps for OptiGenesis.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--images", nargs="+", required=True, help="One or more OCT image paths")
    parser.add_argument("--output-dir", default="figures/paper/gradcam", help="Output directory")
    parser.add_argument("--target-class", type=int, default=1, help="Target class index (default: 1)")
    parser.add_argument(
        "--auto-target",
        action="store_true",
        help="Use model predicted class as CAM target for each image.",
    )
    parser.add_argument("--backbone", default="swin_tiny_patch4_window7_224", help="Backbone name used by checkpoint")
    parser.add_argument(
        "--clinical",
        nargs=3,
        type=float,
        default=[40.0, 1.0, 1.0],
        metavar=("AGE", "HPV", "TCT"),
        help="Clinical triplet used for multi-modal forward",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


class OptiGenesisCamWrapper(nn.Module):
    """Wrap OptiGenesis to accept image-only tensor for CAM."""

    def __init__(self, model: OptiGenesis, clinical_values: List[float], device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "clinical_template",
            torch.tensor(clinical_values, dtype=torch.float32, device=device).view(1, 3),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x5d = x.unsqueeze(1)  # [B,C,H,W] -> [B,1,C,H,W]
        clinical = self.clinical_template.expand(x.shape[0], -1)
        alpha = self.model(x5d, clinical)
        return alpha


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """
    Robust reshape for Swin/Transformer activations:
    - (B, L, C) -> (B, C, H, W)
    - (B, H, W, C) -> (B, C, H, W)
    - (B, C, H, W) -> unchanged
    """
    if tensor.ndim == 4:
        b, d1, d2, d3 = tensor.shape
        # BHWC typical for Swin block internals
        if d3 in (96, 192, 384, 768, 1024):
            return tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.contiguous()

    if tensor.ndim != 3:
        raise ValueError(f"Unexpected activation shape for reshape_transform: {tuple(tensor.shape)}")

    b, n, c = tensor.shape
    h = int(math.sqrt(n))
    if h * h != n:
        h = int(math.floor(math.sqrt(n)))
        while h > 1 and (n % h != 0):
            h -= 1
        w = n // h if h > 0 else n
    else:
        w = h
    return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


def load_model(checkpoint: str, backbone: str, device: torch.device) -> OptiGenesis:
    model = OptiGenesis(model_name=backbone, use_clinical=True, num_classes=2).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_candidate_layers(model: OptiGenesis) -> Dict[str, nn.Module]:
    """
    Candidate layer dict for multi-level CAM sweep.
    Tries best-practice Swin locations with robust fallbacks.
    """
    vb = model.vision_backbone
    candidates: Dict[str, nn.Module] = {}

    # Stage3_LastBlock
    try:
        candidates["Stage3_LastBlock"] = vb.layers[-2].blocks[-1].norm1
    except Exception:
        pass

    # Stage4_FirstBlock
    try:
        candidates["Stage4_FirstBlock"] = vb.layers[-1].blocks[0].norm1
    except Exception:
        pass

    # Stage4_LastBlock
    try:
        candidates["Stage4_LastBlock"] = vb.layers[-1].blocks[-1].norm1
    except Exception:
        pass

    # Safety fallback if model structure differs
    if not candidates:
        try:
            candidates["Backbone_Norm"] = vb.norm
        except Exception:
            raise RuntimeError("No valid candidate layer found in vision_backbone.")

    return candidates


def make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    elif x.ndim == 3 and x.shape[2] > 3:
        x = x[:, :, :3]

    x = x.astype(np.float32)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax > vmin:
        x = (x - vmin) / (vmax - vmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def read_image_rgb(image_path: str) -> np.ndarray:
    # 1) PIL
    try:
        return np.array(Image.open(image_path).convert("RGB"))
    except Exception:
        pass

    # 2) OpenCV
    try:
        bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if bgr is not None:
            if bgr.ndim == 2:
                rgb = cv2.cvtColor(_to_uint8_rgb(bgr), cv2.COLOR_RGB2BGR)
                return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if bgr.ndim == 3 and bgr.shape[2] == 3:
                if bgr.dtype != np.uint8:
                    bgr = _to_uint8_rgb(bgr)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return _to_uint8_rgb(bgr)
    except Exception:
        pass

    # 3) tifffile
    try:
        import tifffile

        arr = tifffile.imread(image_path)
        return _to_uint8_rgb(arr)
    except Exception as e:
        raise RuntimeError(f"Failed to read image '{image_path}' via PIL/OpenCV/tifffile: {e}")


def normalize_cam_percentile(cam_map: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    cam = np.asarray(cam_map, dtype=np.float32)
    lo, hi = np.percentile(cam, low), np.percentile(cam, high)
    if hi <= lo:
        return np.clip(cam, 0.0, 1.0)
    cam = (cam - lo) / (hi - lo)
    return np.clip(cam, 0.0, 1.0)


def render_panel(img_np: np.ndarray, visualization: np.ndarray, title: str, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title("Original OCT Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(visualization)
    axes[1].set_title(title, fontsize=13, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    device = torch.device(args.device)
    transform = make_transform()

    model = load_model(args.checkpoint, args.backbone, device)
    wrapped_model = OptiGenesisCamWrapper(model, args.clinical, device).to(device).eval()
    candidate_layers = get_candidate_layers(model)
    print("[INFO] Candidate layers:")
    for name in candidate_layers:
        print(f"  - {name}")

    for image_path in args.images:
        if not os.path.exists(image_path):
            print(f"[WARN] Skip missing image: {image_path}")
            continue

        rgb_raw = read_image_rgb(image_path)
        img_pil = Image.fromarray(rgb_raw)
        input_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_np = cv2.resize(rgb_raw, (224, 224))
        img_np = np.float32(img_np) / 255.0
        patient_id = os.path.splitext(os.path.basename(image_path))[0]

        # Choose CAM target class
        if args.auto_target:
            with torch.no_grad():
                pred_alpha = wrapped_model(input_tensor)
                pred_class = int(torch.argmax(pred_alpha, dim=1).item())
            target_class = pred_class
        else:
            target_class = args.target_class
        targets = [ClassifierOutputTarget(target_class)]
        print(f"[INFO] {patient_id}: target_class={target_class}")

        for layer_name, target_layer in candidate_layers.items():
            cam = EigenCAM(
                model=wrapped_model,
                target_layers=[target_layer],
                reshape_transform=swin_reshape_transform,
            )
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            grayscale_cam = normalize_cam_percentile(grayscale_cam, low=1.0, high=99.0)
            visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

            safe_layer = layer_name.replace(" ", "_")
            save_path = os.path.join(args.output_dir, f"{patient_id}_gradcam_{safe_layer}.png")
            render_panel(img_np, visualization, f"EigenCAM - {layer_name}", save_path)
            print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    main()
