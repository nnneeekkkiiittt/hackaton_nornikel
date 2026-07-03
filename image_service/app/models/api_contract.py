from pydantic import BaseModel
from typing import List, Optional, Dict


class PreprocessingInfo(BaseModel):
    grayscale: bool
    denoise: bool
    resize: tuple[int, int]
    normalization: str  # e.g. "0-1", "-1-1"


class TensorInfo(BaseModel):
    shape: List[int]
    dtype: str
    values_preview: Optional[List[float]] = None  # для debug


class PredictionInfo(BaseModel):
    label: str
    confidence: float
    metrics: Dict[str, float] = {}

class AnalyzeResponse(BaseModel):
    filename: str

    preprocessing: PreprocessingInfo
    tensor: TensorInfo

    prediction: Optional[PredictionInfo] = None