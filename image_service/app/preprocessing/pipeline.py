from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep

from app.preprocessing.validator import ImageValidator
from app.preprocessing.grayscale import GrayscaleConverter
from app.preprocessing.denoise import Denoiser
from app.preprocessing.contrast import ContrastEnhancer
from app.preprocessing.resize import ImageResizer
from app.preprocessing.normalize import ImageNormalizer
from app.preprocessing.tensor import TensorConverter


class PreprocessingPipeline:

    def __init__(self, target_size: int = 1024):

        self.steps: list[ProcessingStep] = [
            ImageValidator(),
            GrayscaleConverter(),
            Denoiser(),
            ContrastEnhancer(),
            ImageResizer(target_size=target_size),
            ImageNormalizer(),
            TensorConverter(),
        ]

    def process(self, data: ImageData) -> ImageData:

        for step in self.steps:
            data = step.process(data)

        return data