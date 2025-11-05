import os
import sys
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
import torch.nn as nn
import uvicorn

# ✅ Ensure 'src' is in Python path (so model.py can be imported)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import get_model  # after adding src to path


# ---------- FastAPI Setup ----------
app = FastAPI(
    title="Deepfake Detection API",
    description="Detect whether an uploaded face image is REAL or FAKE using EfficientNet-B4",
    version="1.0"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")


# ---------- Load Fine-Tuned Model ----------
model = get_model(num_classes=2)

# Ensure classifier matches training structure (Dropout + Linear)
if hasattr(model.classifier[1], "in_features"):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2)
    )

model = model.to(device)
model.load_state_dict(torch.load("C:/Users/poorv/Desktop/Face-the-Future-Deepfake-ML-Challenge/model_best.pth", map_location=device))
model.eval()

print("✅ Model loaded successfully!")


# ---------- Image Preprocessing ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ---------- API Endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image to get a deepfake prediction.
    Returns label + confidence.
    """
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = torch.softmax(model(image), dim=1)
        label = torch.argmax(preds, dim=1).item()
        confidence = torch.max(preds).item()
        pred_label = "fake" if label == 1 else "real"

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4)
    }


# ---------- Run the App ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
