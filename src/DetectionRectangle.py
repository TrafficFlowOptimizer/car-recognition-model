from pydantic import BaseModel
from Pair import Pair


class DetectionRectangle(BaseModel):
    id: int
    lowerLeft: Pair
    upperRight: Pair
    detected_car_ids = set()
    detected_bus_ids = set()
    # def __init__(self, rectangle_id: int, detection_lower_left: tuple[int, int],
    #              detection_upper_right: tuple[int, int]):
    #     self.rectangle_id = rectangle_id
    #     self.detection_lower_left = detection_lower_left
    #     self.detection_upper_right = detection_upper_right
    #     self.detected_car_ids = set()
    #     self.detected_bus_ids = set()
