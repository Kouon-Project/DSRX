from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from modules.backbones import build_backbone
from modules.duration.codec import DurationGroupCodec


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return (1. - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(0., 0.999).float()


class DurationDDPM(nn.Module):
    """Generate centered group logits and decode them within score budgets."""

    def __init__(
            self,
            codec: DurationGroupCodec,
            timesteps: int = 1000,
            schedule_type: str = 'cosine',
            max_beta: float = 0.01,
            sampling_algorithm: str = 'ddim',
            sampling_steps: int = 20,
            sampling_seed: int | None = 1234,
            backbone_type: str = 'lynxnet2',
            backbone_args: dict | None = None
    ):
        super().__init__()
        if timesteps <= 0:
            raise ValueError('timesteps must be positive.')
        if sampling_steps <= 0:
            raise ValueError('sampling_steps must be positive.')
        if sampling_algorithm not in {'ddim', 'ancestral'}:
            raise ValueError('Duration DDPM supports only ddim and ancestral sampling.')
        self.codec = codec
        self.timesteps = int(timesteps)
        self.sampling_algorithm = sampling_algorithm
        self.sampling_steps = int(sampling_steps)
        self.sampling_seed = sampling_seed
        self.denoise_fn = build_backbone(1, 1, backbone_type, backbone_args or {})

        if schedule_type == 'cosine':
            betas = _cosine_beta_schedule(self.timesteps)
        elif schedule_type == 'linear':
            betas = torch.linspace(1e-4, float(max_beta), self.timesteps)
        else:
            raise ValueError(f'Unsupported duration DDPM schedule: {schedule_type!r}.')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.clamp(min=1e-20))

    @staticmethod
    def _extract(values: Tensor, t: Tensor, target: Tensor) -> Tensor:
        return values.gather(0, t).reshape(t.shape[0], *((1,) * (target.ndim - 1)))

    def _project(self, values: Tensor, ph2word: Tensor) -> Tensor:
        projected = self.codec.project(values[:, 0, 0, :], ph2word)
        return projected[:, None, None, :]

    def _make_noise(
            self,
            condition: Tensor,
            ph2word: Tensor,
            noise: Tensor | None = None,
            seed: int | None = None,
            generator: torch.Generator | None = None
    ) -> Tensor:
        shape = condition.shape[0], 1, 1, condition.shape[1]
        if noise is not None:
            if noise.ndim == 2:
                noise = noise[:, None, None, :]
            if tuple(noise.shape) != shape:
                raise ValueError(f'Duration noise shape must be {shape}, but got {tuple(noise.shape)}.')
            generated = noise.to(device=condition.device, dtype=condition.dtype)
        else:
            if generator is None:
                chosen_seed = self.sampling_seed if seed is None else seed
                if chosen_seed is not None:
                    generator = torch.Generator(device=condition.device).manual_seed(int(chosen_seed))
            generated = torch.randn(
                shape, generator=generator, device=condition.device, dtype=condition.dtype
            )
        return self._project(generated, ph2word)

    def _predict_noise(self, x: Tensor, t: Tensor, condition: Tensor, ph2word: Tensor) -> Tensor:
        prediction = self.denoise_fn(x, t, condition.transpose(1, 2))
        return self._project(prediction, ph2word)

    def _q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        alpha = self._extract(self.alphas_cumprod, t, x_start)
        return alpha.sqrt() * x_start + (1. - alpha).sqrt() * noise

    def _reference_state(self, reference: Tensor, noise: Tensor, timestep: int) -> Tensor:
        if timestep < 0:
            return reference
        t = torch.full(
            (reference.shape[0],), timestep, dtype=torch.long, device=reference.device
        )
        return self._q_sample(reference, t, noise)

    def _prepare_reference(
            self,
            reference_dur: Tensor | None,
            retake_mask: Tensor | None,
            ph2word: Tensor,
            noise: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if reference_dur is None and retake_mask is None:
            return None, None, None
        if reference_dur is None or retake_mask is None:
            raise ValueError('reference_dur and retake_mask must be provided together.')
        reference = self.codec.encode(reference_dur, ph2word)[:, None, None, :]
        group_retake = self.codec.expand_group_mask(retake_mask.bool(), ph2word)
        locked = self.codec.active_mask(ph2word) & ~group_retake
        return reference, locked[:, None, None, :], group_retake

    def _sample_ddim(
            self,
            condition: Tensor,
            ph2word: Tensor,
            noise: Tensor,
            reference: Tensor | None,
            locked: Tensor | None
    ) -> Tensor:
        step_count = min(self.sampling_steps, self.timesteps)
        timesteps = torch.linspace(
            self.timesteps - 1, 0, step_count, device=condition.device
        ).round().long().unique_consecutive()
        x = noise
        first_t = int(timesteps[0].item())
        if reference is not None:
            x = torch.where(locked, self._reference_state(reference, noise, first_t), x)

        for index, current_tensor in enumerate(timesteps):
            current = int(current_tensor.item())
            previous = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            t = torch.full((x.shape[0],), current, dtype=torch.long, device=x.device)
            predicted_noise = self._predict_noise(x, t, condition, ph2word)
            alpha_t = self._extract(self.alphas_cumprod, t, x)
            predicted_start = (x - (1. - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            if previous < 0:
                x = predicted_start
            else:
                alpha_previous = self.alphas_cumprod[previous].to(x)
                x = alpha_previous.sqrt() * predicted_start + (1. - alpha_previous).sqrt() * predicted_noise
            x = self._project(x, ph2word)
            if reference is not None:
                x = torch.where(locked, self._reference_state(reference, noise, previous), x)
        return x

    def _sample_ancestral(
            self,
            condition: Tensor,
            ph2word: Tensor,
            noise: Tensor,
            reference: Tensor | None,
            locked: Tensor | None,
            generator: torch.Generator | None = None
    ) -> Tensor:
        if self.sampling_steps != self.timesteps:
            raise ValueError('Ancestral duration DDPM sampling requires sampling_steps == timesteps.')
        x = noise
        for current in reversed(range(self.timesteps)):
            t = torch.full((x.shape[0],), current, dtype=torch.long, device=x.device)
            predicted_noise = self._predict_noise(x, t, condition, ph2word)
            alpha_t = self._extract(self.alphas_cumprod, t, x)
            predicted_start = (x - (1. - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            if current == 0:
                x = predicted_start
                previous = -1
            else:
                alpha = 1. - self.betas[current]
                alpha_bar = self.alphas_cumprod[current]
                alpha_bar_previous = self.alphas_cumprod[current - 1]
                mean = (
                    self.betas[current] * alpha_bar_previous.sqrt() / (1. - alpha_bar) * predicted_start
                    + (1. - alpha_bar_previous) * alpha.sqrt() / (1. - alpha_bar) * x
                )
                step_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
                x = mean + self.posterior_variance[current].sqrt() * step_noise
                previous = current - 1
            x = self._project(x, ph2word)
            if reference is not None:
                x = torch.where(locked, self._reference_state(reference, noise, previous), x)
        return x

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
            t = torch.randint(0, self.timesteps, (condition.shape[0],), device=condition.device)
            noisy = self._q_sample(target, t, sampled_noise)
            prediction = self._predict_noise(noisy, t, condition, ph2word)
            return {
                'prediction': prediction,
                'target': sampled_noise,
                'active_mask': active_mask,
                't': t,
            }

        chosen_seed = self.sampling_seed if seed is None else seed
        generator = None
        if chosen_seed is not None:
            generator = torch.Generator(device=condition.device).manual_seed(int(chosen_seed))
        sampled_noise = self._make_noise(condition, ph2word, noise=noise, seed=seed, generator=generator)
        reference, locked, group_retake = self._prepare_reference(
            reference_dur, retake_mask, ph2word, sampled_noise
        )
        if self.sampling_algorithm == 'ddim':
            sampled = self._sample_ddim(condition, ph2word, sampled_noise, reference, locked)
        else:
            sampled = self._sample_ancestral(
                condition, ph2word, sampled_noise, reference, locked, generator=generator
            )
        logits = sampled[:, 0, 0, :]
        durations = self.codec.decode(logits, ph2word, word_dur)
        if reference_dur is not None:
            durations = torch.where(
                group_retake,
                durations,
                reference_dur.round().long().to(durations.device)
            )
        return durations
