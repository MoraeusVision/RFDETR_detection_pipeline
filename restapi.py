import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
import uvicorn
from contextlib import asynccontextmanager

from detectors.detector_factory import DetectorFactory
from config_loader import read_from_config
from pipeline_context import FrameContext

CONFIG_PATH = "project/config.json"

# Define lifespan for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    config_path = CONFIG_PATH # replace with your config path
    config = read_from_config(config_path)

    app.state.detector = DetectorFactory.create(
        detector_name=config["detector"],
        model_path=config["model_path"],
        confidence_threshold=config.get("conf", 0.5)
    )
    print("Detector loaded and ready!")

    yield  # the server starts handling requests here

    # --- Shutdown ---
    print("Server shutting down")
    # any cleanup methods can be run here

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/image")
async def upload_image(request: Request):
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Request body must be an image")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Image body is empty")

    image_array = np.frombuffer(body, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Retrieve detector from app.state
    detector = request.app.state.detector

    # Run inference
    ctx = FrameContext(frame=image, timestamp=0)
    ctx = detector.detect(ctx)

    return {
        "detections": [
            {"label": d.label, "confidence": d.confidence, "bbox": list(d.bbox)}
            for d in ctx.detections
        ]
    }

def main():
    uvicorn.run("restapi:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()