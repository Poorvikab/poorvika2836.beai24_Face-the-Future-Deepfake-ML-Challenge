# Face-the-Future-Deepfake-ML-Challenge

# 📌 1. What Is It?

Face-the-Future is a deep learning–based image classification system designed to detect AI-generated (deepfake) facial images.
It uses a fine-tuned EfficientNet-B4 architecture pretrained on ImageNet and adapted to the provided real vs fake dataset.
The model outputs a JSON file with predictions (real / fake) and confidence scores for each test image.

A FastAPI interface is also included for real-time inference — upload an image and get an authenticity prediction instantly.

# ⚙️ 2. Tech Stack & Dependencies

| Component           | Technology Used                          | Purpose                               |
| ------------------- | ---------------------------------------- | ------------------------------------- |
| **Framework**       | PyTorch                                  | Model training and inference          |
| **Model**           | EfficientNet-B4 (pretrained on ImageNet) | Backbone for feature extraction       |
| **Data Handling**   | OpenCV, Torchvision                      | Image processing and augmentation     |
| **Visualization**   | Matplotlib, t-SNE                        | Feature-space inspection and analysis |
| **Deployment**      | FastAPI + Uvicorn                        | REST API for serving predictions      |
| **Version Control** | Git & GitHub                             | Code management                       |
| **Hardware**        | NVIDIA GPU (CUDA)                        | Accelerated fine-tuning               |


# 💾 3. Model File Location

Due to GitHub’s 100 MB file size limit, the trained model weights (model_best.pth, ~300 MB) are stored in Google Drive.

🔗 Download Link:
👉 https://drive.google.com/file/d/1yAswERmNYZwo-LABBKo8LNGy1b7UkHSE/view?usp=sharing (Google Drive)

Once downloaded, place it inside the root project folder, like this:

Face-the-Future-Deepfake-ML-Challenge/
│
├── model_best.pth          ← (place here)
├── src/
├── requirements.txt
├── README.md
└── poorvika2836.beai24.json

# 🧱 4. Folder Structure

After setting up, your project directory should look like this:

Face-the-Future-Deepfake-ML-Challenge/
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── prepare_labels.py
│   └── app.py               # FastAPI app
│                  
├── model_best.pth           # Downloaded model weights
├── poorvika2836.beai24.json # Prediction results
├── requirements.txt
├── poorvika2836_presentation.pptx
└── README.md

# 🚀 5. How to Open & Run the Project

🧩 Clone the Repository

git clone https://github.com/Poorvikab/Face-the-Future-Deepfake-ML-Challenge.git
cd Face-the-Future-Deepfake-ML-Challenge

Create and Activate Virtual Environment

python -m venv venv
venv\Scripts\activate        # (Windows)
# or
source venv/bin/activate     # (macOS/Linux)

Install Requirements

pip install -r requirements.txt


▶️ Run Inference (JSON Predictions)

python src/inference.py
This will read images from data/test/ and produce a JSON output file.

🌐 Run FastAPI (Web Inference)
python src/app.py

Then open your browser and visit:
➡️ http://127.0.0.1:8000/docs


# 📁 6. Project File Descriptions

| File/Folder             | Description                                                                                                                                                                                               
| train.py                | Main training script. Loads and augments the dataset, fine-tunes the EfficientNet-B4 model, and saves the best model as `model_best.pth`.                                                                          
| inference.py            | Generates predictions for the test images using the fine-tuned model. Includes **Test-Time Augmentation (TTA)** for robustness and outputs two JSONs — one for official submission and                              one with confidence scores. 

| app.py                  | A **FastAPI web service** that allows users to upload an image and instantly receive a real/fake prediction with confidence. Runs the trained model as a local API.                                                
| feature_viz.py          | Extracts deep features from the trained model and visualizes them in 2D using **t-SNE** to show how the model separates real vs fake faces in feature space.                                                       
| prepare_labels.py       | Combines individual real/fake label JSON files into a unified `train_labels.json` used for training.                                                                                                               
| model.py                | Defines the **EfficientNet-B4** architecture used for fine-tuning, modifying its classifier layer to output two classes (real/fake).                                                                               
                                                                                                                                                
# 🧩 7. Summary

Goal: Detect AI-generated deepfake images.

Backbone: EfficientNet-B4 (transfer learning + fine-tuning).

Validation Accuracy: ~95.5 %.

Inference Speed: ~150 ms per image (GPU).

Output Format: JSON with prediction + confidence.

Deployment: FastAPI endpoint for real-time testing.
