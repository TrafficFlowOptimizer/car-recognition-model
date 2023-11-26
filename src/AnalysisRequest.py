from typing import List
from DetectionRectangle import DetectionRectangle
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    id: str
    extension: str
    skip_frames: str
    detection_rectangles: List[DetectionRectangle]
    video: str
