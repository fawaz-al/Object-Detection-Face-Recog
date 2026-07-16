"""
Lab 2 - Backend FastAPI untuk layanan deteksi real-time.
Muat model dari Lab 1 (fasterrcnn_person_bicycle.pth) dan expose endpoint /detect.

Jalankan dengan:
    pip install fastapi uvicorn python-multipart torch torchvision pillow
    uvicorn lab2_fastapi_backend:app --reload --port 8000
"""

import io
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = ["background", "person", "bicycle"]
MODEL_PATH = "fasterrcnn_person_bicycle.pth"
SCORE_THRESHOLD = 0.5

app = FastAPI(title="Detection Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(TARGET_CLASSES))
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Bobot dimuat dari {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[WARN] {MODEL_PATH} tidak ditemukan. Jalankan Lab 1 dulu. "
              f"Model akan menggunakan bobot acak sementara.")
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


class Detection(BaseModel):
    class_name: str
    score: float
    box: list  # [x1, y1, x2, y2]


class DetectResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Detection service is running"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = F.to_tensor(img).to(DEVICE)

    with torch.no_grad():
        prediction = model([tensor])[0]

    detections = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < SCORE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(Detection(
            class_name=TARGET_CLASSES[label.item()],
            score=round(score.item(), 4),
            box=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        ))

    return DetectResponse(detections=detections, image_width=img.width, image_height=img.height)
