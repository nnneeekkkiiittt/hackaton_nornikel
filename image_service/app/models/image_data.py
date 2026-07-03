from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ImageData:
    """
    Контейнер, проходящий через весь preprocessing pipeline.
    """

    original_image: np.ndarray
    image: np.ndarray

    filename: str
    path: Path

    width: int
    height: int
    channels: int
    dtype: str

    # физический масштаб (если известен)
    microns_per_pixel: float | None = None

    # сюда постепенно будут складываться результаты этапов
    metadata: dict[str, Any] = field(default_factory=dict)

    tensor: torch.Tensor | None = None