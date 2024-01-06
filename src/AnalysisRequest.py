from typing import List

from pydantic import BaseModel

from DetectionRectangle import DetectionRectangle


class AnalysisRequest(BaseModel):
    id: str
    extension: str
    skip_frames: str
    detection_rectangles: List[DetectionRectangle]
    video: str
