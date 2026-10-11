from .codec import DurationGroupCodec
from .ddpm import DurationDDPM
from .losses import DurationDiffusionLoss, DurationRectifiedFlowLoss
from .reflow import DurationRectifiedFlow
from .sampling import make_indexed_duration_noise


GENERATIVE_DURATION_TYPES = {'ddpm', 'reflow'}
SUPPORTED_DURATION_TYPES = {'deterministic', *GENERATIVE_DURATION_TYPES}


def get_duration_type(config: dict) -> str:
    """Return the configured duration model type while preserving old configs."""
    duration_type = config.get('type', 'deterministic')
    if duration_type not in SUPPORTED_DURATION_TYPES:
        raise ValueError(
            f"Unsupported duration model type: {duration_type!r}. "
            f"Expected one of {sorted(SUPPORTED_DURATION_TYPES)}."
        )
    return duration_type


def get_duration_type_config(config: dict, duration_type: str) -> dict:
    """Read nested model settings, with the flat legacy layout as the default."""
    nested = config.get(duration_type)
    return nested if isinstance(nested, dict) else config


__all__ = [
    'DurationDDPM',
    'DurationDiffusionLoss',
    'DurationGroupCodec',
    'DurationRectifiedFlow',
    'DurationRectifiedFlowLoss',
    'GENERATIVE_DURATION_TYPES',
    'SUPPORTED_DURATION_TYPES',
    'get_duration_type',
    'get_duration_type_config',
    'make_indexed_duration_noise',
]
