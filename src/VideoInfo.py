from pydantic import BaseModel

class VideoInfo(BaseModel):
    id: str
    extension: str
    size: int