# 💳 Credit Scoring AI

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Explore-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-green?style=for-the-badge)

**Predict creditworthiness with precision.** This project leverages advanced Machine Learning algorithms to assess financial risk, processing variables from income to payment history to deliver actionable credit decisions.

---

## 🚀 Project Overview

Financial institutions need robust methods to evaluate loan applicants. This **Credit Scoring Model** mimics real-world scenarios by analyzing:
- **Financial Stability**: Income, Employment Duration.
- **Risk Indicators**: Debt-to-Income Ratio (DTi), Loan Amount.
- **Credit Behavior**: Payment History, Existing Credit Lines.

Using a custom-built synthetic data engine, we model financial profiles and train classifiers to distinguish between low-risk and high-risk candidates.

## 🌟 Key Features

*   **⚡ Synthetic Data Engine**: Generates realistic financial datasets with configurable risk parameters (`data_generator.py`).
*   **📊 Insightful EDA**: Automated Exploratory Data Analysis with correlation heatmaps to visualize risk factors.
*   **🤖 Multi-Model Architecture**: Benchmarks **Logistic Regression**, **Decision Trees**, and **Random Forest**.
*   **📈 Comprehensive Metrics**: Evaluates performance using Precision, Recall, F1-Score, and ROC-AUC.

## 🛠️ Tech Stack

*   **Language**: Python
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-Learn
*   **Visualization**: Matplotlib, Seaborn

## ⚡ Quick Start

### 1. Setup
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Create a fresh synthetic dataset:
```bash
python data_generator.py
```

### 3. Train & Evaluate
Run the training pipeline to see the models in action:
```bash
python eda_and_training.py
```

## 🏆 Results

We tested multiple algorithms to solve the classification problem. **Random Forest** emerged as the champion:

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **~93%** | **0.93** | **0.97** |
| Decision Tree | ~89% | 0.89 | 0.89 |
| Logistic Regression | ~85% | 0.86 | 0.91 |

*The Random Forest model effectively captures non-linear relationships between payment history, debt ratio, and creditworthiness.*

---
*Built for the CodeAlpha Internship.*
