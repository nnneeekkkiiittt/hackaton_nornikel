import grpc
import numpy as np
from PIL import Image
from io import BytesIO
from app import ml_service_pb2, ml_service_pb2_grpc


class MLClient:

    def __init__(self, host: str = "ml-service:50051"):
        self.channel = grpc.insecure_channel(host)
        self.stub = ml_service_pb2_grpc.MLServiceStub(self.channel)

    def predict(self, image: np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255.0).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        pil_image = Image.fromarray(image)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")

        request = ml_service_pb2.PredictRequest(
            image=buffer.getvalue()
        )

        response = self.stub.Predict(request)

        overlay = Image.open(BytesIO(response.overlay_png)).convert("RGB")
        mask = Image.open(BytesIO(response.mask_png))
        return {
            "talc_percentage": response.talc_percentage,
            "overlay": overlay,
            "mask": mask,
            "predicted_class": response.predicted_class,
            "class_probabilities": dict(response.class_probabilities),
        }