# TinyML: Voice and Image on a Microcontroller

Two machine learning models running directly on an Arduino Nano 33 BLE Sense (Cortex-M4), built for the Edge ML course at Åbo Akademi University.

- **Voice: keyword spotting** (`TAMM-project-1_inferencing/`). Recognizes the city names "Oslo" and "Tallinn" in phrases like "weather in Oslo", using MFCC features and a small CNN trained in Edge Impulse. About 80% accuracy across 4 classes (Oslo, Tallinn, noise, unknown).
- **Image: rock-paper-scissors** (`Rock Paper Scissors Game/`). Classifies hand gestures from an OV7670 camera with a CNN running on TensorFlow Lite Micro, without any high-level platform. The model is compressed with pruning and int8 quantization to fit in a 40 KB tensor arena, and keeps up to 92% accuracy on the device.

The full write-up, with all experiments and results, is in [EdgeML_Report.pdf](EdgeML_Report.pdf).

## Contents

| Path | What it is |
|------|------------|
| `TAMM-project-1_inferencing/` | Arduino library exported from Edge Impulse for keyword spotting |
| `Rock Paper Scissors Game/Training_Compression.ipynb` | Training, augmentation, pruning and quantization (Google Colab) |
| `Rock Paper Scissors Game/rock_paper_scissors_inference/` | Arduino sketch: camera capture and on-device inference |
| `Rock Paper Scissors Game/serial_image_viewer.py` | Shows the 32x32 frames sent by the board, for debugging |
| `EdgeML_Report.pdf` | Project report |

## Tools

Arduino IDE, Edge Impulse, TensorFlow / TensorFlow Lite Micro, Google Colab, Python

## Team

Matin Moradi, Tariq Aziz
