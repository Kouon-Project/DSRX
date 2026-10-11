import torch
import torchmetrics
from torch import Tensor

from modules.fastspeech.tts_modules import RhythmRegulator


def duration_shape_checks(pred, target, ph2word, mask=None):
    if mask is None:
        assert pred.shape == target.shape == ph2word.shape, \
            f'shapes of pred, target and ph2word mismatch: {pred.shape}, {target.shape}, {ph2word.shape}'
    else:
        assert pred.shape == target.shape == ph2word.shape == mask.shape, \
            f'shapes of pred, target and mask mismatch: {pred.shape}, {target.shape}, {ph2word.shape}, {mask.shape}'
    assert pred.ndim == 2, f'all inputs should be 2D, but got {pred.shape}'
    assert torch.any(ph2word > 0), 'empty word sequence'
    assert torch.all(ph2word >= 0), 'unexpected negative word index'
    assert ph2word.max() <= pred.shape[1], f'word index out of range: {ph2word.max()} > {pred.shape[1]}'


def linguistic_checks(pred, target, ph2word, mask=None):
    duration_shape_checks(pred, target, ph2word, mask=mask)
    assert torch.all(pred >= 0.), 'unexpected negative ph_dur prediction'
    assert torch.all(target >= 0.), 'unexpected negative ph_dur target'


def _active_mask(ph2word: Tensor, mask: Tensor | None) -> Tensor:
    active = ph2word > 0
    if mask is not None:
        active &= mask.bool()
    return active


def _word_shape(ph2word: Tensor) -> tuple[int, int]:
    return ph2word.shape[0], int(ph2word.max().item()) + 1


def _word_sums(values: Tensor, ph2word: Tensor) -> Tensor:
    return values.new_zeros(*_word_shape(ph2word)).scatter_add(1, ph2word, values)


def _word_counts(ph2word: Tensor, active: Tensor) -> Tensor:
    return torch.zeros(
        *_word_shape(ph2word), dtype=torch.long, device=ph2word.device
    ).scatter_add(1, ph2word, active.long())


def _safe_average(total: Tensor, count: Tensor) -> Tensor:
    return torch.where(
        count > 0,
        total / count.to(dtype=total.dtype),
        torch.full_like(total, float('nan'))
    )


def _iter_multi_phone_indices(ph2word: Tensor, active: Tensor):
    for batch_idx in range(ph2word.shape[0]):
        word_ids = torch.unique(ph2word[batch_idx][active[batch_idx]])
        for word_id in word_ids:
            indices = torch.nonzero(
                active[batch_idx] & (ph2word[batch_idx] == word_id), as_tuple=False
            ).squeeze(1)
            if indices.numel() >= 2:
                yield batch_idx, indices


class RhythmCorrectness(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        assert 0. < tolerance < 1., 'tolerance should be within (0, 1)'
        self.tolerance = tolerance
        self.add_state('correct', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)

        shape = pdur_pred.shape[0], ph2word.max() + 1
        wdur_pred = pdur_pred.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_pred
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        wdur_target = pdur_target.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_target
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        if mask is None:
            wdur_mask = torch.ones_like(wdur_pred, dtype=torch.bool)
        else:
            wdur_mask = mask.new_zeros(*shape).scatter_add(
                1, ph2word, mask
            )[:, 1:].bool()  # [B, T_ph] => [B, T_w]

        correct = torch.abs(wdur_pred - wdur_target) <= wdur_target * self.tolerance
        correct &= wdur_mask

        self.correct += correct.sum()
        self.total += wdur_mask.sum()

    def compute(self) -> Tensor:
        return self.correct / self.total


class PhonemeDurationAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.rr = RhythmRegulator()
        self.add_state('accurate', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.int), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)

        shape = pdur_pred.shape[0], ph2word.max() + 1
        wdur_target = pdur_target.new_zeros(*shape).scatter_add(
            1, ph2word, pdur_target
        )[:, 1:]  # [B, T_ph] => [B, T_w]
        pdur_align = self.rr(pdur_pred, ph2word=ph2word, word_dur=wdur_target)

        accurate = torch.abs(pdur_align - pdur_target) <= pdur_target * self.tolerance
        if mask is not None:
            accurate &= mask

        self.accurate += accurate.sum()
        self.total += pdur_pred.numel() if mask is None else mask.sum()

    def compute(self) -> Tensor:
        return self.accurate / self.total


class WordDurationConstraintViolationRate(torchmetrics.Metric):
    """Rate of word groups whose final phoneme durations break score time."""

    def __init__(self, *, tolerance_frames: float = 0., **kwargs):
        super().__init__(**kwargs)
        assert tolerance_frames >= 0.
        self.tolerance_frames = float(tolerance_frames)
        self.add_state('violated', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        duration_shape_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        finite = torch.isfinite(pdur_pred) & torch.isfinite(pdur_target)
        usable = active & finite
        pred = torch.where(usable, pdur_pred.float(), torch.zeros_like(pdur_pred, dtype=torch.float))
        target = torch.where(usable, pdur_target.float(), torch.zeros_like(pdur_target, dtype=torch.float))
        word_mask = _word_counts(ph2word, active)[:, 1:] > 0
        invalid_words = _word_sums((active & ~finite).long(), ph2word)[:, 1:] > 0
        errors = torch.abs(_word_sums(pred, ph2word)[:, 1:] - _word_sums(target, ph2word)[:, 1:])
        violated = ((errors > self.tolerance_frames) | invalid_words) & word_mask
        self.violated += violated.sum()
        self.total += word_mask.sum()

    def compute(self) -> Tensor:
        return _safe_average(self.violated.float(), self.total)


class WordDurationConstraintMaxError(torchmetrics.Metric):
    """Maximum absolute word-sum error in frames."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            'maximum_error', default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx='max'
        )

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        duration_shape_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        finite = torch.isfinite(pdur_pred) & torch.isfinite(pdur_target)
        usable = active & finite
        pred = torch.where(usable, pdur_pred.float(), torch.zeros_like(pdur_pred, dtype=torch.float))
        target = torch.where(usable, pdur_target.float(), torch.zeros_like(pdur_target, dtype=torch.float))
        word_mask = _word_counts(ph2word, active)[:, 1:] > 0
        invalid_words = _word_sums((active & ~finite).long(), ph2word)[:, 1:] > 0
        errors = torch.abs(_word_sums(pred, ph2word)[:, 1:] - _word_sums(target, ph2word)[:, 1:])
        errors = torch.where(invalid_words, torch.full_like(errors, float('inf')), errors)
        if torch.any(word_mask):
            batch_max = errors[word_mask].max().to(dtype=self.maximum_error.dtype)
            self.maximum_error.copy_(torch.maximum(self.maximum_error, batch_max))

    def compute(self) -> Tensor:
        return self.maximum_error


class InvalidDurationRate(torchmetrics.Metric):
    """Rate of negative, fractional, non-finite, or padding-leaking durations."""

    def __init__(self, *, integer_tolerance: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.integer_tolerance = float(integer_tolerance)
        self.add_state('invalid', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, ph2word: Tensor, mask=None) -> None:
        if mask is None:
            assert pdur_pred.shape == ph2word.shape
        else:
            assert pdur_pred.shape == ph2word.shape == mask.shape
        assert pdur_pred.ndim == 2
        active = _active_mask(ph2word, mask)
        pred = pdur_pred.float()
        finite = torch.isfinite(pred)
        non_integer = finite & ((pred - pred.round()).abs() > self.integer_tolerance)
        invalid_active = active & (~finite | (pred < 0.) | non_integer)
        invalid_padding = ~active & (~finite | (pred.abs() > self.integer_tolerance))
        self.invalid += (invalid_active | invalid_padding).sum()
        self.total += pdur_pred.numel()

    def compute(self) -> Tensor:
        return _safe_average(self.invalid.float(), self.total)


class PhonemeBoundaryMAE(torchmetrics.Metric):
    """Macro-average internal phoneme-boundary error over multi-phone words."""

    def __init__(self, *, frame_scale: float = 1., **kwargs):
        super().__init__(**kwargs)
        assert frame_scale > 0.
        self.frame_scale = float(frame_scale)
        self.add_state('word_error_sum', default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('word_count', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        for batch_idx, indices in _iter_multi_phone_indices(ph2word, active):
            pred_boundaries = torch.cumsum(pdur_pred[batch_idx, indices].float(), dim=0)[:-1]
            target_boundaries = torch.cumsum(pdur_target[batch_idx, indices].float(), dim=0)[:-1]
            word_error = torch.abs(pred_boundaries - target_boundaries).mean() * self.frame_scale
            self.word_error_sum += word_error.to(dtype=self.word_error_sum.dtype)
            self.word_count += 1

    def compute(self) -> Tensor:
        return _safe_average(self.word_error_sum, self.word_count)


class PhonemeBoundaryAccuracy(torchmetrics.Metric):
    """Macro-average internal-boundary accuracy over multi-phone words."""

    def __init__(self, *, tolerance_frames: float, **kwargs):
        super().__init__(**kwargs)
        assert tolerance_frames >= 0.
        self.tolerance_frames = float(tolerance_frames)
        self.add_state('word_accuracy_sum', default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('word_count', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        for batch_idx, indices in _iter_multi_phone_indices(ph2word, active):
            pred_boundaries = torch.cumsum(pdur_pred[batch_idx, indices].float(), dim=0)[:-1]
            target_boundaries = torch.cumsum(pdur_target[batch_idx, indices].float(), dim=0)[:-1]
            accuracy = (
                torch.abs(pred_boundaries - target_boundaries) <= self.tolerance_frames
            ).float().mean()
            self.word_accuracy_sum += accuracy.to(dtype=self.word_accuracy_sum.dtype)
            self.word_count += 1

    def compute(self) -> Tensor:
        return _safe_average(self.word_accuracy_sum, self.word_count)


class DurationAllocationScore(torchmetrics.Metric):
    """One minus total-variation distance between within-word duration ratios."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('score_sum', default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('word_count', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        for batch_idx, indices in _iter_multi_phone_indices(ph2word, active):
            pred = pdur_pred[batch_idx, indices].float()
            target = pdur_target[batch_idx, indices].float()
            target_total = target.sum()
            if target_total <= 0:
                continue
            pred_total = pred.sum()
            if pred_total <= 0:
                score = pred.new_zeros(())
            else:
                total_variation = 0.5 * torch.abs(
                    pred / pred_total - target / target_total
                ).sum()
                score = (1. - total_variation).clamp(min=0., max=1.)
            self.score_sum += score.to(dtype=self.score_sum.dtype)
            self.word_count += 1

    def compute(self) -> Tensor:
        return _safe_average(self.score_sum, self.word_count)


class MultiPhonemeDurationAccuracy(torchmetrics.Metric):
    """Hybrid absolute/relative duration accuracy, excluding trivial one-phone words."""

    def __init__(self, *, relative_tolerance: float = 0.2, absolute_tolerance: float = 2., **kwargs):
        super().__init__(**kwargs)
        assert relative_tolerance >= 0.
        assert absolute_tolerance >= 0.
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.add_state('word_accuracy_sum', default=torch.tensor(0., dtype=torch.float64), dist_reduce_fx='sum')
        self.add_state('word_count', default=torch.tensor(0, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, pdur_pred: Tensor, pdur_target: Tensor, ph2word: Tensor, mask=None) -> None:
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        active = _active_mask(ph2word, mask)
        for batch_idx, indices in _iter_multi_phone_indices(ph2word, active):
            pred = pdur_pred[batch_idx, indices].float()
            target = pdur_target[batch_idx, indices].float()
            tolerance = torch.maximum(
                target * self.relative_tolerance,
                torch.full_like(target, self.absolute_tolerance)
            )
            accuracy = (torch.abs(pred - target) <= tolerance).float().mean()
            self.word_accuracy_sum += accuracy.to(dtype=self.word_accuracy_sum.dtype)
            self.word_count += 1

    def compute(self) -> Tensor:
        return _safe_average(self.word_accuracy_sum, self.word_count)
