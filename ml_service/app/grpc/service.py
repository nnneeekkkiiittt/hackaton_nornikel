from io import BytesIO
import grpc
import numpy as np
from PIL import Image

from app import ml_service_pb2, ml_service_pb2_grpc

from app.inference import TalcPredictor


class MLService(ml_service_pb2_grpc.MLServiceServicer):
    """
    gRPC сервис, оборачивающий TalcPredictor.
    """

    def __init__(self):
        # Загружаем веса ОДИН раз при запуске сервера
        self.predictor = TalcPredictor(
            weights_path="app/best_unet.pth"
        )

    def Predict(self, request, context):
        try:
            # -----------------------------
            # bytes -> PIL.Image
            # -----------------------------
            image = Image.open(BytesIO(request.image)).convert("RGB")

            # PIL -> numpy
            image_np = np.array(image)

            # -----------------------------
            # Инференс
            # -----------------------------
            result = self.predictor.predict_panorama(image_np)

            # -----------------------------
            # overlay -> PNG bytes
            # -----------------------------
            overlay_buffer = BytesIO()
            result["overlay"].save(overlay_buffer, format="PNG")
            overlay_bytes = overlay_buffer.getvalue()

            # -----------------------------
            # mask -> PNG bytes
            # -----------------------------
            mask_buffer = BytesIO()
            result["mask"].save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()

            # -----------------------------
            # Ответ
            # -----------------------------
            return ml_service_pb2.PredictResponse(
                talc_percentage=result["talc_percentage"],
                overlay_png=overlay_bytes,
                mask_png=mask_bytes,
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

            return ml_service_pb2.PredictResponse()