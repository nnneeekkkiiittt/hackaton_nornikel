from abc import ABC, abstractmethod

from app.models.image_data import ImageData


class ProcessingStep(ABC):
    """Base class for all preprocessing steps."""

    @abstractmethod
    def process(self, data: ImageData) -> ImageData:
        pass