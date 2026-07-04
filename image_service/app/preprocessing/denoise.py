import cv2

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class Denoiser(ProcessingStep):
    """
    Removes noise from microscopy images using
    Fast Non-Local Means Denoising.
    """

    def __init__(
        self,
        h: float = 10,
        template_window_size: int = 7,
        search_window_size: int = 21,
    ):
        self.h = h
        self.template_window_size = template_window_size
        self.search_window_size = search_window_size

    def process(self, data: ImageData) -> ImageData:
        data.image = cv2.fastNlMeansDenoising(
            src=data.image,
            h=self.h,
            templateWindowSize=self.template_window_size,
            searchWindowSize=self.search_window_size,
        )

        return data