from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _GroupBalancedGenerativeLoss(nn.Module):
    def __init__(self, loss_type: str = 'l2'):
        super().__init__()
        if loss_type not in {'l1', 'l2'}:
            raise ValueError(f'Unsupported duration generative loss: {loss_type!r}.')
        self.loss_type = loss_type

    def pointwise(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.loss_type == 'l1':
            return torch.abs(prediction - target)
        return torch.square(prediction - target)

    @staticmethod
    def group_mean(error: Tensor, ph2word: Tensor, active_mask: Tensor) -> Tensor:
        if error.ndim != 4 or error.shape[1:3] != (1, 1):
            raise ValueError(f'Expected [B, 1, 1, T] loss tensors, got {tuple(error.shape)}.')
        error = error[:, 0, 0, :]
        if error.shape != ph2word.shape or active_mask.shape != ph2word.shape:
            raise ValueError('Duration loss mask, ph2word and prediction lengths must match.')
        active_mask = active_mask.bool() & (ph2word > 0)
        max_word = int(ph2word.max().item()) if ph2word.numel() else 0
        shape = ph2word.shape[0], max_word + 1
        sums = torch.zeros(shape, dtype=error.dtype, device=error.device).scatter_add(
            1, ph2word, error * active_mask
        )
        counts = torch.zeros(shape, dtype=error.dtype, device=error.device).scatter_add(
            1, ph2word, active_mask.to(error.dtype)
        )
        valid_groups = counts[:, 1:] > 0
        if not torch.any(valid_groups):
            return error.sum() * 0.
        group_losses = sums[:, 1:] / counts[:, 1:].clamp(min=1.)
        return group_losses[valid_groups].mean()


class DurationDiffusionLoss(_GroupBalancedGenerativeLoss):
    def forward(
            self,
            prediction: Tensor,
            target: Tensor,
            ph2word: Tensor,
            active_mask: Tensor
    ) -> Tensor:
        return self.group_mean(self.pointwise(prediction, target), ph2word, active_mask)


class DurationRectifiedFlowLoss(_GroupBalancedGenerativeLoss):
    def __init__(self, loss_type: str = 'l2', log_norm: bool = True):
        super().__init__(loss_type=loss_type)
        self.log_norm = bool(log_norm)

    @staticmethod
    def time_weights(t: Tensor) -> Tensor:
        eps = 1e-7
        t = torch.clip(t.float(), eps, 1. - eps)
        return 0.398942 / t / (1. - t) * torch.exp(
            -0.5 * torch.log(t / (1. - t)) ** 2
        ) + eps

    def forward(
            self,
            prediction: Tensor,
            target: Tensor,
            t: Tensor,
            ph2word: Tensor,
            active_mask: Tensor
    ) -> Tensor:
        error = self.pointwise(prediction, target)
        if self.log_norm:
            error = error * self.time_weights(t)[:, None, None, None]
        return self.group_mean(error, ph2word, active_mask)
