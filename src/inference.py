import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from model import get_model
from tqdm import tqdm


# ---------- Test-Time Augmentation (TTA) ----------
def predict_with_tta(model, image, device):
    """
    Perform Test-Time Augmentation (TTA) for robustness.
    Applies multiple light transformations and averages predictions.
    """
    tta_transforms = [
        lambda x: x,  # original
        lambda x: torch.flip(x, dims=[3]),  # horizontal flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # rotate 90 degrees
        lambda x: torch.rot90(x, 3, [2, 3]),  # rotate -90 degrees
    ]

    preds = []
    with torch.no_grad():
        for t in tta_transforms:
            preds.append(torch.softmax(model(t(image)), dim=1))

    avg_pred = torch.mean(torch.stack(preds), dim=0)
    return avg_pred


# ---------- Inference Pipeline ----------
def generate_predictions(test_dir="data/test"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on {device}...")

    # Load fine-tuned model (same structure as during training)
    model = get_model(num_classes=2)
    if hasattr(model.classifier[1], "in_features"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        )
    model = model.to(device)
    model.load_state_dict(torch.load("model_best.pth", map_location=device))
    model.eval()

    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load test images
    test_imgs = sorted(os.listdir(test_dir))
    print(f"Found {len(test_imgs)} test images")

    results_conf = []  # with confidence
    results_simple = []  # official submission

    with torch.no_grad():
        for i, img_name in enumerate(tqdm(test_imgs, desc="Predicting")):
            img_path = os.path.join(test_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(device)

            # Robust prediction with TTA
            avg_pred = predict_with_tta(model, image, device)
            label = torch.argmax(avg_pred, dim=1).item()
            pred_label = "fake" if label == 1 else "real"
            confidence = torch.max(avg_pred).item()  # Confidence score (0–1)

            # Add both versions
            results_conf.append({
                "index": i + 1,
                "prediction": pred_label,
                "confidence": round(confidence, 4)
            })
            results_simple.append({
                "index": i + 1,
                "prediction": pred_label
            })

    # Save both JSONs
    output_conf = "poorvika2836_with_confidence.json"
    output_simple = "poorvika2836.beai24.json"

    with open(output_conf, "w") as f:
        json.dump(results_conf, f, indent=4)
    with open(output_simple, "w") as f:
        json.dump(results_simple, f, indent=4)

    print(f"✅ Predictions with confidence saved to {output_conf}")
    print(f"✅ Official submission file saved to {output_simple}")


if __name__ == "__main__":
    generate_predictions()

