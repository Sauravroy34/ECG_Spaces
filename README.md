
# ECG Arrhythmia Classifier

A deep learning-powered ECG arrhythmia classification dashboard with **Grad-CAM explainability**.

## Features

- **Upload ECG data** — CSV, TXT, NPY, or paste as JSON array 
- **6-class arrhythmia classification** — ATRIAL, SA, SB, SR, ST, SVT
- **Confidence thresholding** — uncertain predictions are flagged when below threshold
- **1D Grad-CAM visualization** — see which parts of the signal drive the model's decision

## Model

- **Architecture**: ResNet-50 adapted for 1D signals
- **Input shape**: (1, 1, 500) — single-channel, 500-sample ECG
- **Weights**: [Codemaster67/ECG_Arythmia](https://huggingface.co/Codemaster67/ECG_Arythmia)

## Backend Model training code
- **Github repo**: [ResnetBYOL](https://github.com/Sauravroy34/ResnetBYOL)
