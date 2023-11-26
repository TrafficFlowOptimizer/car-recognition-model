from pydantic import BaseModel
from Pair import Pair


class DetectionRectangleRequest(BaseModel):
    id: int
    lowerLeft: Pair
    upperRight: Pair
