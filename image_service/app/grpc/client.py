import grpc
import numpy as np
import torch

from app.grpc import ml_pb2, ml_pb2_grpc


class MLClient:

    def __init__(self, host: str = "ml-service:50051"):
        self.channel = grpc.insecure_channel(host)
        self.stub = ml_pb2_grpc.MLServiceStub(self.channel)

    def predict(self, tensor: torch.Tensor):

        np_tensor = tensor.cpu().numpy()

        request = ml_pb2.TensorRequest(
            data=np_tensor.tobytes(),
            shape=list(np_tensor.shape),
            dtype=str(np_tensor.dtype),
        )

        response = self.stub.Predict(request)

        return {
            "label": response.label,
            "confidence": response.confidence,
        }