import os
import json
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from model import get_model

# ---------- Dataset Class ----------
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


# ---------- Training ----------
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Augmentations for Robustness ---
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(380, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = DeepfakeDataset("data/train", "data/train_labels.json", train_transform)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = get_model(num_classes=2).to(device)

    # Add Dropout for extra regularization (optional)
    if hasattr(model.classifier[1], "in_features"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 2)
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    print(f"Training on {device} with {len(train_dataset)} train and {len(val_dataset)} val images...")

    best_val_acc = 0
    for epoch in range(13):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # ---- Validation ----
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = torch.argmax(model(imgs), dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "model_best.pth")
            print("✅ Model improved and saved as model_best.pth")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_model()
