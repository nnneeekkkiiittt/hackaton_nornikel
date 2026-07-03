import torch
import numpy as np

from app.models.image_data import ImageData
from app.preprocessing.base import ProcessingStep


class TensorConverter(ProcessingStep):

    def process(self, data: ImageData) -> ImageData:

        image = data.image

        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image)
        else:
            tensor = image

        # HWC → CHW
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)

        # add batch dim
        tensor = tensor.unsqueeze(0)

        data.tensor = tensor

        return data