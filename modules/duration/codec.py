from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DurationGroupCodec(nn.Module):
    """Convert phoneme durations to and from group-normalized duration logits.

    The score-provided word duration remains a hard constraint. Generative
    models only sample how that duration is distributed among the phonemes in
    each word. Single-phoneme words bypass generation entirely.
    """

    def __init__(
            self,
            smoothing_frames: float = 0.5,
            logit_scale: float = 2.0,
            logit_clip: float = 5.0,
            min_frames: int = 1
    ):
        super().__init__()
        if smoothing_frames <= 0:
            raise ValueError('smoothing_frames must be positive.')
        if logit_scale <= 0:
            raise ValueError('logit_scale must be positive.')
        if logit_clip <= 0:
            raise ValueError('logit_clip must be positive.')
        if min_frames < 0:
            raise ValueError('min_frames must not be negative.')
        self.smoothing_frames = float(smoothing_frames)
        self.logit_scale = float(logit_scale)
        self.logit_clip = float(logit_clip)
        self.min_frames = int(min_frames)

    @staticmethod
    def _group_shape(ph2word: Tensor) -> tuple[int, int]:
        if ph2word.ndim != 2:
            raise ValueError(f'ph2word must be 2-D, but got {tuple(ph2word.shape)}.')
        max_word = int(ph2word.max().item()) if ph2word.numel() else 0
        return ph2word.shape[0], max_word + 1

    @classmethod
    def group_counts(cls, ph2word: Tensor) -> Tensor:
        batch, groups = cls._group_shape(ph2word)
        valid = ph2word > 0
        return torch.zeros(batch, groups, dtype=torch.long, device=ph2word.device).scatter_add(
            1, ph2word, valid.long()
        )

    @classmethod
    def active_mask(cls, ph2word: Tensor) -> Tensor:
        counts = cls.group_counts(ph2word)
        per_phone_counts = torch.gather(counts, 1, ph2word)
        return (ph2word > 0) & (per_phone_counts > 1)

    @classmethod
    def expand_group_mask(cls, mask: Tensor, ph2word: Tensor) -> Tensor:
        if mask.shape != ph2word.shape:
            raise ValueError(
                f'Group mask and ph2word shapes differ: {tuple(mask.shape)} vs {tuple(ph2word.shape)}.'
            )
        batch, groups = cls._group_shape(ph2word)
        group_mask = torch.zeros(batch, groups, dtype=torch.long, device=ph2word.device).scatter_add(
            1, ph2word, (mask & (ph2word > 0)).long()
        ) > 0
        return torch.gather(group_mask, 1, ph2word) & (ph2word > 0)

    @classmethod
    def word_durations(cls, ph_dur: Tensor, ph2word: Tensor) -> Tensor:
        if ph_dur.shape != ph2word.shape:
            raise ValueError(
                f'ph_dur and ph2word shapes differ: {tuple(ph_dur.shape)} vs {tuple(ph2word.shape)}.'
            )
        batch, groups = cls._group_shape(ph2word)
        durations = ph_dur.float() * (ph2word > 0)
        return torch.zeros(batch, groups, dtype=durations.dtype, device=durations.device).scatter_add(
            1, ph2word, durations
        )[:, 1:]

    @classmethod
    def project(cls, values: Tensor, ph2word: Tensor) -> Tensor:
        """Project values onto the zero-mean subspace of multi-phone groups."""
        if values.shape != ph2word.shape:
            raise ValueError(
                f'Values and ph2word shapes differ: {tuple(values.shape)} vs {tuple(ph2word.shape)}.'
            )
        batch, groups = cls._group_shape(ph2word)
        active = cls.active_mask(ph2word)
        values = values * active
        sums = torch.zeros(batch, groups, dtype=values.dtype, device=values.device).scatter_add(
            1, ph2word, values
        )
        counts = torch.zeros(batch, groups, dtype=values.dtype, device=values.device).scatter_add(
            1, ph2word, active.to(values.dtype)
        )
        means = sums / counts.clamp(min=1.)
        return (values - torch.gather(means, 1, ph2word)) * active

    def encode(self, ph_dur: Tensor, ph2word: Tensor) -> Tensor:
        """Encode absolute frame durations as standardized centered log-ratios."""
        if ph_dur is None:
            raise ValueError('Ground-truth or reference ph_dur is required for duration encoding.')
        if ph_dur.shape != ph2word.shape:
            raise ValueError(
                f'ph_dur and ph2word shapes differ: {tuple(ph_dur.shape)} vs {tuple(ph2word.shape)}.'
            )
        batch, groups = self._group_shape(ph2word)
        valid = ph2word > 0
        durations = ph_dur.float()
        counts = self.group_counts(ph2word).to(durations.dtype)
        totals = torch.zeros(batch, groups, dtype=durations.dtype, device=durations.device).scatter_add(
            1, ph2word, durations * valid
        )
        phone_counts = torch.gather(counts, 1, ph2word)
        phone_totals = torch.gather(totals, 1, ph2word)
        probabilities = (durations + self.smoothing_frames) / (
            phone_totals + self.smoothing_frames * phone_counts
        ).clamp(min=self.smoothing_frames)
        logits = torch.log(probabilities.clamp(min=1e-8))
        logits = self.project(logits, ph2word) / self.logit_scale
        return logits.clamp(min=-self.logit_clip, max=self.logit_clip)

    def _group_probabilities(self, logits: Tensor, ph2word: Tensor) -> Tensor:
        active = self.active_mask(ph2word)
        logits = self.project(
            logits.clamp(min=-self.logit_clip, max=self.logit_clip),
            ph2word
        ) * self.logit_scale
        scores = torch.exp(logits.clamp(min=-20., max=20.)) * active
        batch, groups = self._group_shape(ph2word)
        group_sums = torch.zeros(batch, groups, dtype=scores.dtype, device=scores.device).scatter_add(
            1, ph2word, scores
        )
        probabilities = scores / torch.gather(group_sums, 1, ph2word).clamp(min=1e-8)
        return probabilities * active

    def decode(self, logits: Tensor, ph2word: Tensor, word_dur: Tensor) -> Tensor:
        """Decode logits to integer phoneme frames with exact group sums."""
        if logits.shape != ph2word.shape:
            raise ValueError(
                f'Logit and ph2word shapes differ: {tuple(logits.shape)} vs {tuple(ph2word.shape)}.'
            )
        if word_dur is None or word_dur.ndim != 2:
            raise ValueError('word_dur must be a [B, T_word] tensor during generative inference.')
        batch, groups_with_padding = self._group_shape(ph2word)
        word_count = groups_with_padding - 1
        if word_dur.shape[0] != batch or word_dur.shape[1] < word_count:
            raise ValueError(
                f'word_dur shape {tuple(word_dur.shape)} cannot cover {word_count} word groups.'
            )

        probabilities = self._group_probabilities(logits.float(), ph2word)
        result = torch.zeros_like(ph2word, dtype=torch.long)
        rounded_word_dur = word_dur.round().long().clamp(min=0)

        # Allocate whole frames, then distribute the remainder within each group.
        for batch_idx in range(batch):
            for word_idx in range(1, word_count + 1):
                indices = torch.nonzero(ph2word[batch_idx] == word_idx, as_tuple=False).squeeze(1)
                if indices.numel() == 0:
                    continue
                total = int(rounded_word_dur[batch_idx, word_idx - 1].item())
                phone_count = int(indices.numel())
                if phone_count == 1:
                    result[batch_idx, indices[0]] = total
                    continue

                floor_per_phone = self.min_frames if total >= self.min_frames * phone_count else 0
                remaining = total - floor_per_phone * phone_count
                phone_probabilities = probabilities[batch_idx, indices]
                probability_sum = phone_probabilities.sum()
                if not torch.isfinite(probability_sum) or probability_sum <= 0:
                    phone_probabilities = torch.full_like(phone_probabilities, 1. / phone_count)
                else:
                    phone_probabilities = phone_probabilities / probability_sum

                raw = phone_probabilities * remaining
                allocation = torch.floor(raw).long() + floor_per_phone
                remainder = total - int(allocation.sum().item())
                if remainder > 0:
                    order = torch.argsort(raw - torch.floor(raw), descending=True)
                    full_cycles, tail = divmod(remainder, phone_count)
                    if full_cycles:
                        allocation += full_cycles
                    if tail:
                        allocation[order[:tail]] += 1
                result[batch_idx, indices] = allocation

        return result
