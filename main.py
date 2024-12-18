from fastapi import FastAPI, File, UploadFile
import cv2
import torch
import numpy as np
import dlib
import torch.nn.functional as F
from fastapi.responses import JSONResponse

# Load model and detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name="se_resnext50_32x4d", num_classes=101):
    import pretrainedmodels
    import torch.nn as nn

    model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def load_model():
    model = get_model()
    path = "./pretrained.pth"  # Ensure this file is in the container
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


model = load_model()
face_detector = dlib.get_frontal_face_detector()

app = FastAPI()


def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    detected = face_detector(image, 1)
    faces = []

    for d in detected:
        x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
        face = cv2.resize(image[y1:y2, x1:x2], (224, 224))
        faces.append(face)

    if not faces:
        return None

    inputs = torch.from_numpy(
        np.transpose(np.array(faces).astype(np.float32), (0, 3, 1, 2))
    ).to(device)
    outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
    ages = np.arange(0, 101)
    predicted_ages = (outputs * ages).sum(axis=-1)
    return int(predicted_ages[0])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        age = process_image(image_bytes)
        if age is None:
            return JSONResponse(content={"error": "No face detected"}, status_code=400)
        return {"age": age}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
