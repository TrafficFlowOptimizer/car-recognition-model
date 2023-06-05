from fastapi import FastAPI
import requests
from uuid import uuid4
import os
import glob
from time import sleep

from VideoInfo import VideoInfo

# from Model import Model

app = FastAPI()

VIDEOS_DIRECTORY = "../vids"

SERVER = "http://localhost:8080/"

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
    
    print("Starting download: {} ({}B)".format(file_name, video_info.size))
    
    with open(file_name, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                video_file.write(chunk)

    print("Video {} successfully saved".format(video_info.id))
    
    # model = Model(file_name)
    # model.run()
    
    await delete_video(file_name)
    
    return "in the future car flow here"