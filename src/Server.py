from fastapi import FastAPI
import requests
from uuid import uuid4

from VideoInfo import VideoInfo

# from Model import Model

app = FastAPI()

SERVER = "http://localhost:8080/"

VIDEOS = "videos/"
OPTIMIZATIONS = "optimization/"

@app.get("/")
async def home():
    return "Hello world!"

@app.post("/analysis")
async def analysis_request(video_info: VideoInfo) -> str: # for now string; later proper object with car flow
    response = requests.get(SERVER + VIDEOS + video_info.id, stream=True)
    response.raise_for_status()
    
    file_name = "../vids/vid" + video_info.id + "_" + str(uuid4()) + "." + video_info.extension
    
    print("Starting download: {} ({}B)".format(file_name, video_info.size))
    
    with open(file_name, "wb") as video_file:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                video_file.write(chunk)

    print("Video {} successfully saved".format(video_info.id))
    
    # model = Model(file_name)
    # model.run()
    
    return "here"