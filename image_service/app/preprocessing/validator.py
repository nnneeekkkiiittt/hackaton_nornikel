from __future__ import annotations

import numpy as np

from app.models.image_data import ImageData

from app.preprocessing.base import ProcessingStep


class ImageValidationError(Exception):
    """Raised when image validation fails."""


class ImageValidator(ProcessingStep):
    """
    Validates an ImageData object before preprocessing.
    """

    SUPPORTED_DTYPES = {
        np.uint8,
        np.uint16,
        np.float32,
    }

    MIN_WIDTH = 128
    MIN_HEIGHT = 128

    def process(self, data: ImageData) -> ImageData:
        image = data.image

        if image is None:
            raise ImageValidationError("Image is None.")

        if not isinstance(image, np.ndarray):
            raise ImageValidationError(
                f"Expected numpy.ndarray, got {type(image)}."
            )

        if image.size == 0:
            raise ImageValidationError("Image is empty.")

        if image.ndim not in (2, 3):
            raise ImageValidationError(
                f"Unsupported image dimensions: {image.ndim}."
            )

        if image.ndim == 3:
            channels = image.shape[2]

            if channels not in (1, 3, 4):
                raise ImageValidationError(
                    f"Unsupported number of channels: {channels}."
                )

        if image.dtype.type not in self.SUPPORTED_DTYPES:
            raise ImageValidationError(
                f"Unsupported dtype: {image.dtype}"
            )

        if image.shape[0] < self.MIN_HEIGHT:
            raise ImageValidationError(
                f"Image height ({image.shape[0]}) is too small."
            )

        if image.shape[1] < self.MIN_WIDTH:
            raise ImageValidationError(
                f"Image width ({image.shape[1]}) is too small."
            )

        if not np.isfinite(image).all():
            raise ImageValidationError(
                "Image contains NaN or Inf values."
            )

        return data