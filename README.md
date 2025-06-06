# Brain Tumor Classification using ResNet-18

## Overview

1. Uses a ResNet-18 model with PyTorch to classify brain MRI images.
2. Designed to outperform a previous YOLOv8-CLS implementation.
3. `main.py` — Training script.
4. `aug.py` — Data augmentation utilities.
5. `predict.py` — Streamlit app for interactive prediction.

## Notes

1. Pretrained `.pth` model is included, so `predict.py` can be run directly.
2. Grad-CAM will be added in the future to visualize regions of interest for better explainability.
3. Future updates may include transformer-based architectures.

## Requirements

1. Python 3.11
2. CUDA-enabled GPU is recommended for training the model from scratch.

## Dataset

* [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

