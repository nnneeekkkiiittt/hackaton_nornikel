import cv2

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class GrayscaleConverter(ProcessingStep):
    """
    Converts an RGB/RGBA image to grayscale.
    If the image is already grayscale, leaves it unchanged.
    """

    def process(self, data: ImageData) -> ImageData:
        image = data.image

        if image.ndim == 2:
            return data

        channels = image.shape[2]

        if channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        data.image = image
        data.channels = 1
        data.height, data.width = image.shape

        return data