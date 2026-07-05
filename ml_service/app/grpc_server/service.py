from io import BytesIO
import grpc
import numpy as np
from PIL import Image

from app import ml_service_pb2, ml_service_pb2_grpc

from app.inference import TalcPredictor, ClassPredictor


class MLService(ml_service_pb2_grpc.MLServiceServicer):
    """
    gRPC сервис, оборачивающий TalcPredictor и ClassPredictor.
    """

    def __init__(self):
        # Загружаем веса ОДИН раз при запуске сервера
        self.predictor = TalcPredictor(
            weights_path="app/best_unet.pth"
        )
        self.class_predictor = ClassPredictor(
            weights_path="app/best_classifier_weights.pt"
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
            class_result = self.class_predictor.predict(image_np)
            
            # Эвристика: если классификатор уверен на 85%+, что это не оталькованная руда,
            # задираем порог обнаружения талька до 0.99, чтобы убрать ложные срабатывания.
            probs = class_result["probabilities"]
            not_talcose_prob = probs.get("ordinary", 0.0) + probs.get("refractory", 0.0)
            
            talc_thresh = 0.85
            if not_talcose_prob >= 0.85:
                talc_thresh = 0.999
                
            print(f"[DEBUG] Ore classification: {class_result['class']} (probs: {probs}), Not-talcose prob: {not_talcose_prob:.4f} -> threshold set to: {talc_thresh}", flush=True)
                
            result = self.predictor.predict_panorama(image_np, talc_threshold=talc_thresh)

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
                predicted_class=class_result["class"],
                class_probabilities=class_result["probabilities"]
            )

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

            return ml_service_pb2.PredictResponse()