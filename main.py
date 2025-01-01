from fastapi import FastAPI, File, UploadFile
import cv2
import torch
import numpy as np
import dlib
import torch.nn.functional as F
from fastapi.responses import JSONResponse
import ssl

# Load model and detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name="se_resnext50_32x4d", num_classes=101):
    import pretrainedmodels
    import torch.nn as nn

    # Disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context

    model = pretrainedmodels.__dict__[model_name](pretrained="imagenet")
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model


def load_model():
    model = get_model()
    path = "./pretrained.pth"  # Ensure this file is in the container
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


model = load_model()
face_detector = dlib.get_frontal_face_detector()

app = FastAPI()


def process_image(image_bytes):
    input_size = 224
    margin = 0.4
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    image_h, image_w = image.shape[:2]
    detected = face_detector(image, 1)
    faces = np.empty((len(detected), input_size, input_size, 3))

    for d in detected:
        x1, y1, x2, y2, w, h = (
            d.left(),
            d.top(),
            d.right() + 1,
            d.bottom() + 1,
            d.width(),
            d.height(),
        )
        xw1 = max(int(x1 - margin * w), 0)
        yw1 = max(int(y1 - margin * h), 0)
        xw2 = min(int(x2 + margin * w), image_w - 1)
        yw2 = min(int(y2 + margin * h), image_h - 1)
        faces[i] = cv2.resize(
            image[yw1 : yw2 + 1, xw1 : xw2 + 1], (input_size, input_size)
        )

    if not faces:
        return None

    inputs = torch.from_numpy(
        np.transpose(np.array(faces).astype(np.float32), (0, 3, 1, 2))
    ).to(device)
    outputs = F.softmax(model(inputs), dim=-1).detach().cpu().numpy()
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
