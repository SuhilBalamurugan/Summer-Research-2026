# Summer-Research-2026

# Isolating Emotion from Motion: Classifying Affective States from Wearable Physiological Signals

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Dataset: WESAD](https://img.shields.io/badge/Dataset-WESAD-green.svg)](https://ubicomp.eti.uni-siegen.de/home/datasets/wesad/)

This repository hosts the code, experimental pipeline, and evaluation suite for investigating whether machine learning classifiers trained on wearable physiological signals rely on genuine autonomic physiology or incidental body motion to distinguish emotional states.

---

## Project Overview
Wearable emotion and stress detection systems increasingly rely on optical and bio-potential sensors, which are notoriously susceptible to motion artifacts. In controlled laboratory environments, participants are typically seated, introducing the risk that classifiers inadvertently learn motion-correlated shortcuts rather than true physiological markers.

This project investigates this question using the multimodal **WESAD** (Wearable Stress and Affect Detection) dataset under strict, subject-independent validation.

### Core Objectives
1. **Evaluate Affect Classification:** Classify three affective conditions (neutral baseline, stress, and amusement) using chest-worn physiological telemetry.
2. **Test the Motion Confound:** Directly test whether classifier predictions depend on body motion through explicit feature-ablation experiments and tree-based feature-importance analyses.

---

## Dataset & Preprocessing
Experiments were conducted using chest-worn physiological recordings from the **WESAD** benchmark dataset (15 subjects, RespiBAN chest unit sampled at 700 Hz):
- **Signals:** Electrocardiography (ECG), Electrodermal Activity (EDA), Respiration (RESP), Peripheral Skin Temperature (TEMP), and 3-axis Accelerometry (ACC).
- **Segmentation:** Continuous 700 Hz signals segmented into non-overlapping 60-second windows.
- **Feature Extraction:** 14 statistical features per window (mean, standard deviation, extrema) across modalities. Two collinear EDA metrics ($r = 1.00$) were ablated to avoid singularity.
- **Dataset Size:** 520 labeled windows across 15 participants:
  - **Baseline:** 283 windows (54.4%)
  - **Stress:** 155 windows (29.8%)
  - **Amusement:** 82 windows (15.8%)
  - *Class Imbalance Ratio:* ~3.4:1

---

## Methodology
- **Validation Framework:** **Leave-One-Subject-Out (LOSO) Cross-Validation**. In each of the 15 folds, the model is trained on 14 participants and tested exclusively on the remaining held-out subject, preventing identity leakage and estimating true generalization to unseen users.
- **Primary Metric:** **Macro-F1 score** (unweighted per-class average). Under substantial class imbalance, accuracy is inflated by majority-class bias.
- **Class Balancing:** Class weights inversely proportional to class frequencies were applied across all models.
- **Classifiers Evaluated:**
  - **Logistic Regression:** Interpretable, low-variance linear baseline.
  - **Random Forest (300 trees):** Non-linear ensemble providing Gini impurity-based feature rankings.
  - **Support Vector Machine (RBF Kernel):** Standard benchmark for physiological affective classification.

---

## Key Experimental Results

### 1. Model Comparison (LOSO Cross-Validation)
Logistic Regression demonstrated the strongest class-balanced performance, successfully balancing minority and majority classes.

| Model | LOSO Accuracy | Macro-F1 Score |
| :--- | :---: | :---: |
| **Logistic Regression** | **0.648** | **0.623** |
| **SVM (RBF Kernel)** | 0.629 | 0.616 |
| **Random Forest (300 trees)** | 0.731 | 0.577 |

> **Why Accuracy Misleads:** Random Forest achieved the highest raw accuracy (0.731) but the worst macro-F1 (0.577) because it over-predicted the majority baseline class, collapsing to an amusement recall of only 0.07. Macro-F1 exposed this failure mode.

### 2. Per-Class Performance (Logistic Regression)
- **Stress** is detected reliably ($F_1 = 0.74$, $\text{Recall} = 0.79$) due to clear autonomic arousal in heart rate and skin conductance.
- **Amusement** remains the most challenging state ($F_1 = 0.46$, $\text{Precision} = 0.37$), physiologically overlapping with quiet baseline rest in time-domain metrics (74 baseline windows misclassified as amusement).

| Affective State | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Baseline** | 0.79 | 0.58 | 0.67 |
| **Stress** | 0.71 | 0.79 | **0.74** |
| **Amusement** | 0.37 | 0.61 | 0.46 |

### 3. Testing the Motion Confound: Feature Ablation vs. Feature Importance
To assess whether models relied on incidental movement as a classification shortcut, accelerometer features (`ACC_mean`, `ACC_std`) were removed:

| Model | Full Features (Macro-F1) | Motion Ablated (Macro-F1) | $\Delta$ Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 0.623 | **0.640** | +0.017 |
| **Random Forest** | 0.577 | **0.612** | +0.035 |
| **SVM (RBF)** | 0.616 | **0.632** | +0.016 |

- **The Paradox:** Random Forest ranked `ACC_std` as its 2nd most important feature overall on the complete feature set.
- **The Reconciliation:** `ACC_std` is strongly collinear with peripheral temperature variability (`Temp_std`, $r = 0.94$). When motion features are removed, correlated physiological features retain equivalent predictive variance with zero degradation in performance.
- **Takeaway:** **Motion is used when available, but not needed.** The models classify affective state using genuine autonomic physiology rather than motion shortcuts.

---

## Repository Structure
```text
├── data/
│   └── raw/                       # Instructions for acquiring raw WESAD dataset
├── notebooks/
│   ├── 01_eda_and_feature_distribution.ipynb
│   └── 02_loso_evaluation_and_ablation.ipynb
├── src/
│   ├── data_loader.py             # WESAD parser and signal windowing
│   ├── feature_extraction.py      # Statistical feature extraction
│   ├── models.py                  # LR, RF, and SVM definitions
│   └── evaluate.py                # LOSO cross-validation engine
├── figures/                       # Confusion matrices and ablation charts
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md
