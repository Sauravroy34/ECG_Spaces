---
title: ECG Arrhythmia Classifier
emoji: ❤️
colorFrom: red
colorTo: purple
sdk: streamlit
sdk_version: "1.45.1"
app_file: app.py
pinned: false
license: mit
---

# ECG Arrhythmia Classifier

A deep learning-powered ECG arrhythmia classification dashboard with **Grad-CAM explainability**.

## Features

- **Upload ECG data** — CSV, TXT, NPY, or paste as JSON array (500 data points)
- **6-class arrhythmia classification** — ATRIAL, SA, SB, SR, ST, SVT
- **Confidence thresholding** — uncertain predictions are flagged when below threshold
- **1D Grad-CAM visualization** — see which parts of the signal drive the model's decision

## Model

- **Architecture**: ResNet-50 adapted for 1D signals
- **Input shape**: (1, 1, 500) — single-channel, 500-sample ECG
- **Weights**: [Codemaster67/ECG_Arythmia](https://huggingface.co/Codemaster67/ECG_Arythmia)

## Usage

Upload a file containing 500 numeric values representing a single-lead ECG recording.
The app will display:
1. ECG signal preview
2. Predicted class with confidence score
3. Probability distribution across all classes
4. Grad-CAM heatmap overlay for model interpretability
