import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import get_model


# ---------- Dataset ----------
class DeepfakeDataset(Dataset):
    def __init__(self, base_dir, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.labels = json.load(f)
        self.base_dir = base_dir
        self.transform = transform
        self.samples = []
        for item in self.labels:
            idx = item["index"]
            label = item["prediction"]
            folder = "real" if label == "real" else "fake"
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = os.path.join(base_dir, folder, f"{idx}{ext}")
                if os.path.exists(img_path):
                    self.samples.append((img_path, 0 if label == "real" else 1))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------- Setup ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Extracting features using {device}...")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

dataset = DeepfakeDataset("data/train", "data/train_labels.json", transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# ---------- Load Model (matching training structure) ----------
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

# ---------- Feature Extraction ----------
features, labels_all = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        # Extract intermediate CNN features
        feats = model.features(imgs).mean([2, 3]).cpu().numpy()
        features.append(feats)
        labels_all.extend(labels.numpy())

X = np.concatenate(features)
y = np.array(labels_all)
print(f"Extracted feature shape: {X.shape}")

# ---------- Dimensionality Reduction ----------
print("Running t-SNE (this may take a minute)...")
X_embedded = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

# ---------- Visualization ----------
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], label='Real', alpha=0.6, c='green')
plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], label='Fake', alpha=0.6, c='red')
plt.legend()
plt.title("Feature Space Visualization (t-SNE) - Real vs Fake Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.tight_layout()
plt.show()
