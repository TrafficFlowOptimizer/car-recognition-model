from fastapi import FastAPI
import uvicorn
import requests
from uuid import uuid4
import os
from sys import argv
import glob
from time import sleep
from subprocess import Popen, PIPE

from VideoInfo import VideoInfo

# from Model import Model

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
async def home(): return "Hello world!"

@app.post("/analysis")
async def analysis_request(video_info: VideoInfo) -> str: # for now string; later proper object with car flow return
    response = requests.get(SERVER + VIDEOS + video_info.id, stream=True)
    response.raise_for_status()
    
    file_name = VIDEOS_DIRECTORY + "/vid" + video_info.id + "_" + str(uuid4()) + "." + video_info.extension
    
    print("Starting download: {}".format(file_name))
    
    with open(file_name, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                video_file.write(chunk)

    print("Video {} successfully saved".format(video_info.id))
    
    # analysis_process = Popen(["python", "car_counter_yolov3_COCO_6_classes.py", "-y", "yolo", "--input", "videos/asia.mp4", "--output", "output", "--skip-frames", "5"])
    # res = analysis_process.communicate()[0]
    # print(res)
    # print(analysis_process.returncode == 0)
    
    await delete_video(file_name)
    
    return "in the future car flow here"

if __name__ == "__main__":
    uvicorn.run("Server:app", port=8081, host="0.0.0.0", reload=True)