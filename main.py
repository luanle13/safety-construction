from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple
from PIL import Image
import numpy as np
import cv2
import io
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os

app = FastAPI()

repo_id = "ltl1313ltl/construction-safety"
model_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

model = YOLO(model_path)

LABEL_MAP = {
    1: 'no-helmet',
    2: 'no-vest',
}

origins = [
    "http://localhost:3000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(os.getcwd(), "static")), name="static")

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img_height, img_width = image_bgr.shape[:2]

    results = model.predict(image_bgr)

    bounding_boxes = []
    confidence_threshold = 0.5

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            confidence = box.conf.tolist()[0]
            label = box.cls.tolist()[0]

            if confidence > confidence_threshold and label in LABEL_MAP:
                label_name = LABEL_MAP.get(label)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{label_name} - {confidence}"

                cv2.putText(
                    image_bgr,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

                bounding_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": label,
                    "class_name": label_name
                })
    total_count = len(bounding_boxes)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output_image_dir = "static/images"
    os.makedirs(output_image_dir, exist_ok=True)  # Ensure output directory exists
    output_image_path = os.path.join(output_image_dir, "output_image.jpg")
    cv2.imwrite(output_image_path, image_rgb)

    return {
        "message": "Image processed successfully",
        "image_url": f"/static/images/output_image.jpg",
        "bounding_boxes": bounding_boxes,
        "total_count": total_count
    }
                