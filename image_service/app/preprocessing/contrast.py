import cv2

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class ContrastEnhancer(ProcessingStep):
    """
    Enhances local contrast using CLAHE.
    """

    def process(self, data: ImageData) -> ImageData:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        data.image = clahe.apply(data.image)

        return data