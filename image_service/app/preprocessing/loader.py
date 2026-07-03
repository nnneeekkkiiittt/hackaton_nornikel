from pathlib import Path
from app.models.image_data import ImageData

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