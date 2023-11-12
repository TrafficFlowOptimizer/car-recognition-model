from pydantic import BaseModel
from Pair import Pair


class DetectionRectangle(BaseModel):
    connectionId: int
    lowerLeft: Pair
    upperRight: Pair
    detected_car_ids = set()
    detected_bus_ids = set()
