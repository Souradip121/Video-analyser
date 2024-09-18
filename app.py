from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import requests
import base64
import os
from openai import OpenAI
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)
# Define input model
class VideoInput(BaseModel):
    url: str

# Helper function to download video
def download_video(url: str):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    temp_video.write(chunk)
                return temp_video.name
        else:
            raise Exception("Failed to download video")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Helper function to extract frames from video
def extract_frames(video_path: str, frame_interval: int = 50):
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    frame_count = 0
    
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1
    
    video.release()
    return base64Frames

# Route to process video and generate description
@app.post("/generate-description/")
async def generate_description(video_input: VideoInput):
    # Step 1: Download the video from the provided URL
    video_path = download_video(video_input.url)

    # Step 2: Extract frames from the downloaded video
    base64_frames = extract_frames(video_path)

    # Step 3: Prepare the prompt for OpenAI
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "These are frames from a video.Generate a detailed description of the video. The video will be given by user as a means to report grievance related to train or station to Rail Assist platform of India",
                *map(lambda x: {"image": x, "resize": 768}, base64_frames[0::50]),
            ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }

    # Step 4: Call OpenAI to get the video description
    try:
        result = client.chat.completions.create(**params)
        description = result.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with OpenAI API: {str(e)}")

    # Step 5: Return the generated description
    return {"description": description}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
