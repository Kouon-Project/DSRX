from collections.abc import Iterable

import torch
from torch import Tensor


def make_indexed_duration_noise(
        indices: Iterable[int] | Tensor,
        length: int,
        *,
        base_seed: int,
        device: torch.device | str
) -> Tensor:
    """Create batch-order-independent validation noise for duration sampling."""
    rows = []
    max_seed = 2 ** 63 - 1
    for index in indices:
        sample_index = int(index.item()) if isinstance(index, Tensor) else int(index)
        generator = torch.Generator(device='cpu')
        generator.manual_seed((int(base_seed) + sample_index) % max_seed)
        rows.append(torch.randn(length, generator=generator, dtype=torch.float32))
    if not rows:
        return torch.empty((0, length), dtype=torch.float32, device=device)
    return torch.stack(rows, dim=0).to(device=device)
