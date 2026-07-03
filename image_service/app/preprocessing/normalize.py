import numpy as np

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class ImageNormalizer(ProcessingStep):

    def process(self, data: ImageData) -> ImageData:

        image = data.image.astype(np.float32)

        # 0–255 → 0–1
        image = image / 255.0

        data.image = image

        return data