from fastapi import APIRouter, File, UploadFile, HTTPException

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

        data = loader.from_bytes(
            image_bytes,
            filename=file.filename,
        )

        result = pipeline.process(data)
        import base64
        from io import BytesIO

        prediction = ml_client.predict(result.image)

        buffered_overlay = BytesIO()
        prediction["overlay"].save(buffered_overlay, format="PNG")
        overlay_base64 = base64.b64encode(buffered_overlay.getvalue()).decode("utf-8")

        buffered_mask = BytesIO()
        prediction["mask"].save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "filename": result.filename,
            "shape": str(result.image.shape),
            "dtype": str(result.image.dtype),
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