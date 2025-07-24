# Anomaly-Detection-SS25

Colab notebook + code for UNM crowd anomaly detection

# UMN Anomaly Detection

A lightweight Colab pipeline for frame‐level anomaly detection on UMN Crowd videos using:

- **Frame extraction** → JPGs
- **Captioning** (ViT-GPT2)
- **Anomaly scoring** (FLAN-T5-small & Dolly-v2-3b)
- **Visual diagnostics** (time-series, histograms, ROC/PR, F1 vs threshold, calibration)

## 🚀 Quick Start

1. Open `umn_anomaly_detection.ipynb` in Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run all cells

## 📁 Structure

├── data/ # source AVI clips

├── frames/ # extracted .jpg frames

├── captions/ # per‐frame .txt captions

├── scores/ # CSVs of (frame,score) per model

└── plots/ # generated .png summaries

Feel free to adjust sampling rate, prompts or models in the notebook!

[... Farhan Zahin ...]
