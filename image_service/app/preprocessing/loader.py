import tempfile
from pathlib import Path
from app.models.image_data import ImageData
import numpy as np
import cv2

from app.preprocessing.base import ProcessingStep


class ImageLoader(ProcessingStep):
    """
    Загружает изображение с диска.

    Не изменяет изображение.
    Не выполняет preprocessing.
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    }

    def process(self, path: str | Path) -> ImageData:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(
                f"Cannot open image: {path}"
            )

        if image.ndim == 2:
            channels = 1
            height, width = image.shape

        else:
            height, width, channels = image.shape

        return ImageData(
            image=image,
            filename=path.name,
            path=path,
            width=width,
            height=height,
            channels=channels,
            dtype=str(image.dtype),
        )

    def from_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> ImageData:
        # decode bytes → numpy image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Cannot decode image from bytes")

        # shape parsing
        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape

        path = Path(filename)

        return ImageData(
            original_image=image.copy(),
            image=image,
            filename=path.name,
            path=path,
            width=width,
            height=height,
            channels=channels,
            dtype=str(image.dtype),
            microns_per_pixel=None,
            metadata={}
        )