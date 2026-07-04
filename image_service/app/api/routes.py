from fastapi import APIRouter, File, UploadFile, HTTPException

from app.preprocessing.loader import ImageLoader
from app.preprocessing.pipeline import PreprocessingPipeline
from app.grpc.client import MLClient

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
        prediction = ml_client.predict(result.image)
        return {
            "status": "success",
            "filename": result.filename,
            "talc_percentage": prediction["talc_percentage"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )