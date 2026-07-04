from fastapi import APIRouter, File, UploadFile, HTTPException
import base64
from io import BytesIO
from app.preprocessing.loader import ImageLoader
from app.preprocessing.pipeline import PreprocessingPipeline
from app.grpc_client.client import MLClient

router = APIRouter(prefix="/api")

loader = ImageLoader()
pipeline = PreprocessingPipeline()
ml_client = MLClient()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        from PIL import Image
        from io import BytesIO
        import numpy as np
        
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
        
        prediction = ml_client.predict(image_np)

        buffered_overlay = BytesIO()
        prediction["overlay"].save(buffered_overlay, format="PNG")
        overlay_base64 = base64.b64encode(buffered_overlay.getvalue()).decode("utf-8")

        buffered_mask = BytesIO()
        prediction["mask"].save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "filename": file.filename,
            "talc_percentage": prediction["talc_percentage"],
            "predicted_class": prediction["predicted_class"],
            "class_probabilities": prediction["class_probabilities"],
            "overlay": overlay_base64,
            "mask": mask_base64,
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )