import json
import subprocess

from fastapi import FastAPI
import uvicorn
from uuid import uuid4
import os
from sys import argv
import glob
from subprocess import Popen

from AnalysisRequest import AnalysisRequest

app = FastAPI()

if len(argv) == 1:
    SERVER = "http://localhost:8080/"
    VIDEOS_DIRECTORY = "vids"
else: #Server.py runs on docker
    SERVER = "http://host.docker.internal:8080/"
    VIDEOS_DIRECTORY = "../vids"
    
VIDEOS = "videos/"
OPTIMIZATIONS = "optimization/"

async def delete_video(name): os.remove(name)

async def clear_videos():
    for f in glob.glob(VIDEOS_DIRECTORY): os.remove(f)

@app.get("/")
def home(): return "Hello world!"

@app.post("/analysis")
def analysis_request(analysis_request: AnalysisRequest) -> str: # for now string; later proper object with car flow return
    # response = requests.get(SERVER + VIDEOS + video_info.id, stream=True)
    # response.raise_for_status()
    
    file_name = VIDEOS_DIRECTORY + VIDEOS + analysis_request.id + "_" + str(uuid4()) + "." + analysis_request.extension
    
    print("Starting download: {}".format(file_name))
    
    # with open(file_name, "wb") as video_file:
    #     for chunk in response.iter_content(chunk_size=4096):
    #         if chunk:
    #             video_file.write(chunk)

    print("Video {} successfully saved".format(analysis_request.id))
    
    analysis_process = Popen(["python", "car_counter_yolov3_COCO_6_classes.py", "-y", "yolo", "--input",
                              VIDEOS + analysis_request.id + '.' + analysis_request.extension, "--output",
                              "output", "--skip-frames", analysis_request.skip_frames, "--detection_rectangles",
                              analysis_request.detection_rectangles], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = analysis_process.communicate()

    stdout = stdout.decode('utf-8')

    print('Standard Output:')
    print(stdout)

    # res = analysis_process.communicate()[0]
    # print(res)
    # print(analysis_process.returncode == 0)

    # await delete_video(file_name)
    return stdout

if __name__ == "__main__":
    uvicorn.run("Server:app", port=8081, host="0.0.0.0", reload=True)