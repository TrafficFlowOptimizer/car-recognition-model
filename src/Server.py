import json
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
import os
from sys import argv
import glob

from AnalysisRequest import AnalysisRequest
from src.DetectionRectangle import DetectionRectangle
from src.car_counter_yolov3_COCO_6_classes import CarCounter

app = FastAPI()

# SERVER = os.getenv('SPRING_HOST')

if len(argv) == 1:
    SERVER = "http://localhost:8080/"
    VIDEOS_DIRECTORY = "vids"
else:  # Server.py runs on docker
    SERVER = "http://host.docker.internal:8080/"
    VIDEOS_DIRECTORY = "../vids"

load_dotenv("../.env")
VIDEOS = os.getenv('VIDEOS')
OPTIMIZATIONS = os.getenv('OPTIMIZATIONS')
PORT = int(os.getenv('PORT'))


def parse_detection_rectangles(detection_rectangles_string) -> list[DetectionRectangle]:
    detection_rectangles_parsed_json = json.loads(detection_rectangles_string)
    detection_rectangles = []
    for detection_rectangle in detection_rectangles_parsed_json:
        detection_rectangles.append(DetectionRectangle(detection_rectangle["id"], detection_rectangle["lower_left"],
                                                       detection_rectangle["upper_right"]))
    return detection_rectangles


async def delete_video(name): os.remove(name)


async def clear_videos():
    for f in glob.glob(VIDEOS_DIRECTORY): os.remove(f)


@app.get("/")
def home(): return "Hello world!"


@app.post("/analysis")
def analysis_request(
        analysis_request: AnalysisRequest) -> str:

    car_counter = CarCounter("yolo", VIDEOS + analysis_request.id + '.' + analysis_request.extension,
                             "output", int(analysis_request.skip_frames),
                             analysis_request.detection_rectangles, analysis_request.video)

    count_cars = car_counter.run()

    return count_cars


if __name__ == "__main__":
    uvicorn.run("Server:app", port=PORT, host="0.0.0.0", reload=True)
