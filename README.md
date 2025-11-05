# Face-the-Future-Deepfake-ML-Challenge

📌 1. What Is It?

Face-the-Future is a deep learning–based image classification system designed to detect AI-generated (deepfake) facial images.
It uses a fine-tuned # EfficientNet-B4 architecture pretrained # on ImageNet and adapted to the provided real vs fake dataset.
The model outputs a JSON file with predictions (real / fake) and confidence scores for each test image.

A FastAPI interface is also included for real-time inference — upload an image and get an authenticity prediction instantly.
