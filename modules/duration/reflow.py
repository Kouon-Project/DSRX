from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from modules.backbones import build_backbone
from modules.duration.codec import DurationGroupCodec


class DurationRectifiedFlow(nn.Module):
    """Pure generative rectified-flow duration allocator."""

    def __init__(
            self,
            codec: DurationGroupCodec,
            time_scale_factor: float = 1000.,
            sampling_algorithm: str = 'rk2',
            sampling_steps: int = 8,
            sampling_seed: int | None = 1234,
            backbone_type: str = 'lynxnet2',
            backbone_args: dict | None = None
    ):
        super().__init__()
        if time_scale_factor <= 0:
            raise ValueError('time_scale_factor must be positive.')
        if sampling_steps <= 0:
            raise ValueError('sampling_steps must be positive.')
        if sampling_algorithm not in {'euler', 'rk2', 'rk4'}:
            raise ValueError('Duration ReFlow supports euler, rk2 and rk4 sampling.')
        self.codec = codec
        self.time_scale_factor = float(time_scale_factor)
        self.sampling_algorithm = sampling_algorithm
        self.sampling_steps = int(sampling_steps)
        self.sampling_seed = sampling_seed
        self.velocity_fn = build_backbone(1, 1, backbone_type, backbone_args or {})

    def _project(self, values: Tensor, ph2word: Tensor) -> Tensor:
        projected = self.codec.project(values[:, 0, 0, :], ph2word)
        return projected[:, None, None, :]

    def _make_noise(
            self,
            condition: Tensor,
            ph2word: Tensor,
            noise: Tensor | None = None,
            seed: int | None = None
    ) -> Tensor:
        shape = condition.shape[0], 1, 1, condition.shape[1]
        if noise is not None:
            if noise.ndim == 2:
                noise = noise[:, None, None, :]
            if tuple(noise.shape) != shape:
                raise ValueError(f'Duration noise shape must be {shape}, but got {tuple(noise.shape)}.')
            generated = noise.to(device=condition.device, dtype=condition.dtype)
        else:
            chosen_seed = self.sampling_seed if seed is None else seed
            if chosen_seed is None:
                generated = torch.randn(shape, device=condition.device, dtype=condition.dtype)
            else:
                generator = torch.Generator(device=condition.device)
                generator.manual_seed(int(chosen_seed))
                generated = torch.randn(
                    shape, generator=generator, device=condition.device, dtype=condition.dtype
                )
        return self._project(generated, ph2word)

    def _velocity(self, x: Tensor, t: Tensor, condition: Tensor, ph2word: Tensor) -> Tensor:
        prediction = self.velocity_fn(
            x,
            t * self.time_scale_factor,
            condition.transpose(1, 2)
        )
        return self._project(prediction, ph2word)

    @staticmethod
    def _reference_state(reference: Tensor, noise: Tensor, t: float) -> Tensor:
        return noise + t * (reference - noise)

    def _step(self, x: Tensor, t: float, dt: float, condition: Tensor, ph2word: Tensor) -> Tensor:
        time = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
        if self.sampling_algorithm == 'euler':
            x = x + self._velocity(x, time, condition, ph2word) * dt
        elif self.sampling_algorithm == 'rk2':
            k1 = self._velocity(x, time, condition, ph2word)
            middle_time = torch.full((x.shape[0],), t + dt * 0.5, dtype=x.dtype, device=x.device)
            k2 = self._velocity(x + k1 * (dt * 0.5), middle_time, condition, ph2word)
            x = x + k2 * dt
        else:
            k1 = self._velocity(x, time, condition, ph2word)
            half_time = torch.full((x.shape[0],), t + dt * 0.5, dtype=x.dtype, device=x.device)
            end_time = torch.full((x.shape[0],), t + dt, dtype=x.dtype, device=x.device)
            k2 = self._velocity(x + k1 * (dt * 0.5), half_time, condition, ph2word)
            k3 = self._velocity(x + k2 * (dt * 0.5), half_time, condition, ph2word)
            k4 = self._velocity(x + k3 * dt, end_time, condition, ph2word)
            x = x + (k1 + 2. * k2 + 2. * k3 + k4) * (dt / 6.)
        return self._project(x, ph2word)

    def forward(
            self,
            condition: Tensor,
            ph2word: Tensor,
            ph_dur: Tensor | None = None,
            word_dur: Tensor | None = None,
            infer: bool = True,
            noise: Tensor | None = None,
            seed: int | None = None,
            reference_dur: Tensor | None = None,
            retake_mask: Tensor | None = None
    ):
        active_mask = self.codec.active_mask(ph2word)
        if not infer:
            target = self.codec.encode(ph_dur, ph2word)[:, None, None, :].to(condition.dtype)
            sampled_noise = self._project(torch.randn_like(target), ph2word)
            t = torch.rand((condition.shape[0],), device=condition.device, dtype=condition.dtype)
            noisy = sampled_noise + t[:, None, None, None] * (target - sampled_noise)
            prediction = self._velocity(noisy, t, condition, ph2word)
            return {
                'prediction': prediction,
                'target': target - sampled_noise,
                'active_mask': active_mask,
                't': t,
            }

        sampled_noise = self._make_noise(condition, ph2word, noise=noise, seed=seed)
        reference = locked = group_retake = None
        if reference_dur is not None or retake_mask is not None:
            if reference_dur is None or retake_mask is None:
                raise ValueError('reference_dur and retake_mask must be provided together.')
            reference = self.codec.encode(reference_dur, ph2word)[:, None, None, :]
            group_retake = self.codec.expand_group_mask(retake_mask.bool(), ph2word)
            locked = (self.codec.active_mask(ph2word) & ~group_retake)[:, None, None, :]

        x = sampled_noise
        dt = 1. / self.sampling_steps
        for step in range(self.sampling_steps):
            t = step * dt
            x = self._step(x, t, dt, condition, ph2word)
            if reference is not None:
                x = torch.where(
                    locked,
                    self._reference_state(reference, sampled_noise, (step + 1) * dt),
                    x
                )
        durations = self.codec.decode(x[:, 0, 0, :], ph2word, word_dur)
        if reference_dur is not None:
            durations = torch.where(
                group_retake,
                durations,
                reference_dur.round().long().to(durations.device)
            )
        return durations
