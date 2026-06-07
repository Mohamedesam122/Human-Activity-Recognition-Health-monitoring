# 🧠 Human Activity Recognition — End-to-End Deep Learning System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

**A production-level AI system for classifying human physical activities from smartphone sensor data using classical machine learning, deep learning, attention mechanisms, and explainable AI.**

[Live Demo](#-dashboard) · [Dataset](#-dataset) · [Models](#-models) · [Results](#-results)

</div>

---

## 📌 Overview

This project builds a complete pipeline for **Human Activity Recognition (HAR)** using the UCI HAR Dataset — raw accelerometer and gyroscope signals collected from 30 subjects performing 6 daily activities.

The system covers three progressive phases:

| Phase | Focus | Key Methods |
|-------|-------|-------------|
| **Phase 1** | Classical ML | Feature Engineering · KNN · Naive Bayes · Random Forest · Gradient Boosting · PCA · LDA |
| **Phase 2** | Deep Learning | 1D-CNN · Training Optimization · Hyperparameter Tuning |
| **Phase 3** | Advanced AI | LSTM · BiLSTM · Attention-LSTM · Transformer · Conditional GAN · SHAP |

---

## 🏃 Activities Recognized

| Activity | Label |
|----------|-------|
| 🚶 Walking | `WALKING` |
| ⬆️ Walking Upstairs | `WALKING_UPSTAIRS` |
| ⬇️ Walking Downstairs | `WALKING_DOWNSTAIRS` |
| 🪑 Sitting | `SITTING` |
| 🧍 Standing | `STANDING` |
| 🛌 Laying | `LAYING` |

---

## 📊 Dataset

**UCI HAR Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/jorgeromn/human-activity-recognition-smartphones-data-set)

| Property | Value |
|----------|-------|
| Subjects | 30 |
| Train samples | 7,352 |
| Test samples | 2,947 |
| Sensor channels | 9 (acc x/y/z · gyro x/y/z · total acc x/y/z) |
| Timesteps per sample | 128 |
| Sampling rate | 50 Hz |
| Window duration | 2.56 seconds |

**Sensor signals:**
- `body_acc_x/y/z` — Body acceleration
- `body_gyro_x/y/z` — Body angular velocity
- `total_acc_x/y/z` — Total acceleration including gravity

---

## 🏗️ Project Structure

```
HAR-Project/
│
├── project.ipynb              ← Main notebook (Phase 1 + 2 + 3)
├── app.py                     ← Streamlit interactive dashboard
├── requirements.txt           ← Python dependencies
├── README.md
│
├── data/
│   └── UCI HAR Dataset/       ← Raw dataset
│
└── models/                    ← Saved trained models (after running notebook)
    ├── rf_model.pkl
    ├── scaler.pkl
    ├── cnn_model.h5
    ├── lstm_model.h5
    ├── attn_model.h5
    └── transformer_model.h5
```

---

## ⚙️ Feature Engineering

From each 128-timestep window across 9 sensor channels, **90 features** are extracted:

**Time-domain (63 features):** mean · std · min · max · median · skewness · kurtosis

**Frequency-domain (27 features):** FFT dominant frequency · FFT energy · spectral entropy

---

## 🤖 Models

### Phase 1 — Classical ML

| Model | Accuracy |
|-------|----------|
| K-Nearest Neighbors | ~88% |
| Naive Bayes | ~77% |
| Random Forest | ~92% |
| Gradient Boosting | ~93% |

### Phase 2 — Convolutional Neural Network

```
Input (128 × 9)
→ Conv1D(64, k=5) → BatchNorm → ReLU → MaxPool(2)
→ Conv1D(128, k=3) → BatchNorm → ReLU → MaxPool(2)
→ Conv1D(256, k=3) → BatchNorm → ReLU → GlobalAvgPool
→ Dense(128) → Dropout(0.4)
→ Dense(64) → Dropout(0.3)
→ Dense(6, softmax)
```

### Phase 3 — Sequential & Advanced Models

**LSTM:**
```
Input → LSTM(128, return_sequences=True) → Dropout(0.3)
→ LSTM(64) → Dropout(0.3) → Dense(64) → Dense(6, softmax)
```

**Bidirectional LSTM:**
```
Input → BiLSTM(128, return_sequences=True) → Dropout(0.3)
→ BiLSTM(64) → Dropout(0.3) → Dense(64) → Dense(6, softmax)
```

**Attention-LSTM (Bahdanau):**
```
Input → LSTM(128, return_sequences=True)
→ BahdanauAttention(64) → weighted context vector
→ Dense(64) → Dropout(0.3) → Dense(6, softmax)
```

**Transformer Encoder:**
```
Input → Linear Projection(64) → Positional Encoding
→ [MultiHeadAttention(4 heads) → LayerNorm → FFN → LayerNorm] × 2
→ GlobalAvgPool → Dense(64) → Dropout(0.3) → Dense(6, softmax)
```

**Conditional GAN** — Generates synthetic sensor signals for minority activity classes to handle class imbalance.

---

## 📈 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | ~93% | ~93% |
| 1D-CNN | ~93% | ~93% |
| LSTM | ~92% | ~92% |
| BiLSTM | ~93% | ~93% |
| Attention-LSTM | ~93% | ~93% |
| Transformer | ~93% | ~93% |

---

## 🔍 Explainable AI — SHAP

SHAP (SHapley Additive exPlanations) is applied to the best Phase 1 tree model to explain:

- **Global feature importance** — which sensor features matter most across all predictions
- **Per-sensor channel importance** — which of the 9 channels contributes most
- **Local explanation** — why the model predicted a specific activity for one sample
- **Feature dependence plots** — how individual feature values affect predictions
- **Waterfall plots** — single-sample prediction breakdown

---

## 🖥️ Dashboard

An interactive Streamlit dashboard (`app.py`) with 6 pages:

| Page | Description |
|------|-------------|
| 🎯 Prediction | Select any activity → real sample → all models predict → confidence scores |
| 📡 Sensor Signals | Interactive raw signal plots for all 9 channels with FFT |
| ⚙️ Feature Engineering | Extracted features visualization · FFT spectrum · bar charts |
| 🔥 Attention Heatmap | Attention weights over 128 timesteps per activity |
| 📊 SHAP Explainability | Global & local SHAP explanations + per-channel importance |
| 🏆 Model Comparison | Accuracy · F1 · Precision · Recall + radar chart |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HAR-Project.git
cd HAR-Project
```

### 2. Create virtual environment

```bash
python -m venv har_env
har_env\Scripts\activate      # Windows
source har_env/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download the UCI HAR Dataset from [Kaggle](https://www.kaggle.com/datasets/jorgeromn/human-activity-recognition-smartphones-data-set) and place it in the project folder as `UCI HAR Dataset/`.

Or run the notebook — it downloads automatically via `kagglehub`.

### 5. Run the notebook

Open `project.ipynb` in Jupyter and run all cells top to bottom.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow==2.15.0
numpy==1.26.4
pandas
scikit-learn
scipy
streamlit
plotly
shap
matplotlib
seaborn
mediapipe==0.10.7
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.15 · Keras |
| Classical ML | Scikit-learn |
| Signal Processing | NumPy · SciPy |
| Explainability | SHAP |
| Dashboard | Streamlit · Plotly |
| Visualization | Matplotlib · Seaborn |

---

## 📚 References

- Davide Anguita et al. — *A Public Domain Dataset for Human Activity Recognition Using Smartphones* (2013)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate* (2015)
- Vaswani et al. — *Attention Is All You Need* (2017)
- Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions* (2017)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Built for AI Engineer Portfolio · Healthcare AI · IoT + Deep Learning

⭐ Star this repo if you found it useful

</div>
