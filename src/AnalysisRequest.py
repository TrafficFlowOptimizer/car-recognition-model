from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    id: str
    extension: str
    skip_frames: str
    detection_rectangles: str