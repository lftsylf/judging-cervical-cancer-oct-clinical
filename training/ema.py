"""
指数滑动平均（EMA）权重，用于稳定训练、常作为轻量泛化正则。

用法：
- 每个 optimizer.step() 之后调用 update(model)
- 验证 / 选模 / 存盘前 apply_shadow(model)，结束后 restore(model)
"""

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module) -> None:
        self.backup.clear()
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = param.data.clone()
            param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
