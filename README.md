# 🌡️ Non-Contact Facial Analysis Using Infrared Face Images

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

## 📌 Overview

Infrared imaging can capture facial information without touching the patient, which makes it attractive for contactless screening in clinics, elder care and public spaces.

This project builds a complete deep learning pipeline for infrared face images: preprocessing, face detection, CNN classification and Grad-CAM explainability. As a first proof of concept, the pipeline is trained on a **binary facial occlusion task (mask vs no mask)**, which tests whether the models can learn meaningful facial features from low-contrast infrared images. The same pipeline is designed to be retrained on labelled health condition data.

## 🔄 Pipeline

```
Infrared images ─> Pairing with RGB ─> CLAHE enhancement ─> Face detection + crop ─> CNN / DenseNet-121 ─> Prediction + Grad-CAM heatmap
```

## 🗂️ Dataset

- **Source:** `[ADD: dataset name and link]`
- **Size:** 7 subjects × 2 conditions (mask, no mask) × 5 head poses = **70 infrared images**, each paired with an RGB photo
- **Poses:** head up, head down, head turned left, head turned right, head rolled right
- **Split:** 80% train (56) / 20% validation (14), stratified by label

## 🧪 Methodology

### 1. Data Preparation
- Parse the folder structure into a manifest (subject, mask status, image type, pose)
- Pair each infrared image with its RGB counterpart by subject, condition and pose

### 2. Preprocessing
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to bring out detail in low-contrast infrared images
- Haar cascade face detection, keeping the largest detected face
- Center crop fallback when no face is detected
- Resize to 224 × 224

### 3. Models
**Custom CNN (baseline)**
- 3 convolutional blocks (16, 32, 64 filters) with ReLU and max pooling
- Fully connected layer (128 units) with dropout 0.3

**DenseNet-121 (transfer learning)**
- Pretrained on ImageNet, classifier head replaced for 2 classes
- Fine-tuned end to end, best checkpoint saved on validation accuracy

**Training setup:** Adam (learning rate 1e-4), cross-entropy loss, 10 epochs, batch size 16, random flip and rotation augmentation

### 4. Explainability
- **Grad-CAM** on DenseNet's final feature layer, showing which facial regions drove each prediction

## 📊 Results

| Model | Best Validation Accuracy |
|-------|--------------------------|
| Custom CNN | 71.43% |
| DenseNet-121 (transfer learning) | **85.71%** |

**DenseNet-121 classification report (validation set, n = 14)**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Mask | 0.86 | 0.86 | 0.86 |
| No mask | 0.86 | 0.86 | 0.86 |

Transfer learning clearly helped: the pretrained DenseNet learned useful features from just 56 training images, where the custom CNN struggled.

> ⚠️ The validation set has only 14 images, and the split is by image rather than by subject, so the same people appear in both training and validation. These results show the pipeline works, not how it would perform on new people.

## 🔭 Future Work

- Retrain on infrared data labelled for health conditions (e.g. fever, inflammation, circulation changes)
- Use thermal temperature values rather than image intensity alone
- Subject-wise splits and cross-validation to test on unseen people
- Larger and more diverse dataset

## 📁 Project Structure

```
├── Non_Contact_Facial_Analysis_Infrared.ipynb
├── models/
│   ├── healthnet_infrared.pth
│   └── densenet_infrared.pth
├── manifests/
│   ├── infrared_manifest.csv
│   └── processed_ir_manifest.csv
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

The notebook was built in Google Colab with the dataset stored on Google Drive. To run locally, replace the Drive paths with your local dataset path.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook
```

**requirements.txt**
```
torch
torchvision
opencv-python
pandas
numpy
matplotlib
seaborn
scikit-learn
pillow
tqdm
```

## 🛠️ Tech Stack

| Area | Tools |
|------|-------|
| Language | Python |
| Image processing | OpenCV (CLAHE, Haar cascades), Pillow |
| Deep learning | PyTorch, torchvision (DenseNet-121) |
| Explainability | Grad-CAM |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |

## 👤 Author

**Sushmitha K**
Always learning, always building at the edge of medicine and machines.
