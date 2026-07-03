from fastapi import APIRouter, File, UploadFile, HTTPException

from app.preprocessing.loader import ImageLoader
from app.preprocessing.pipeline import PreprocessingPipeline

router = APIRouter(prefix="/api")

loader = ImageLoader()
pipeline = PreprocessingPipeline()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        data = loader.from_bytes(
            image_bytes,
            filename=file.filename,
        )

        result = pipeline.process(data)

        return {
            "status": "success",
            "filename": result.filename,
            "shape": list(result.tensor.shape),
            "dtype": str(result.tensor.dtype),
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )