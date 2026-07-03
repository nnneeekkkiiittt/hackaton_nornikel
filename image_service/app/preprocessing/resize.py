import cv2
import numpy as np

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class ImageResizer(ProcessingStep):

    def __init__(self, target_size: int = 1024):
        self.target_size = target_size

    def process(self, data: ImageData) -> ImageData:

        image = data.image

        h, w = image.shape[:2]

        scale = min(
            self.target_size / w,
            self.target_size / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.zeros(
            (self.target_size, self.target_size),
            dtype=resized.dtype,
        )

        x = (self.target_size - new_w) // 2
        y = (self.target_size - new_h) // 2

        canvas[
            y:y + new_h,
            x:x + new_w
        ] = resized

        data.image = canvas
        data.width = self.target_size
        data.height = self.target_size

        return data