# CodeAlpha — ML Project Portfolio

> Intelligent, efficient, and reproducible ML experiments for credit scoring and medical disease prediction.

## Overview

CodeAlpha bundles small, production-minded machine learning projects that showcase data engineering, exploratory data analysis, model training, and reproducible pipelines.

## Included Projects

- **CodeAlpha_Credit_Scoring_Model** — A full pipeline for credit risk scoring using tabular loan data, including EDA, feature engineering, and model training. See [CodeAlpha_Credit_Scoring_Model/README.md](CodeAlpha_Credit_Scoring_Model/README.md) for details.
- **CodeAlpha_DiseasePrediction_from_Medical_data** — End-to-end disease prediction from clinical data: preprocessing, model definitions, and evaluation. See [CodeAlpha_DiseasePrediction_from_Medical_data/README.md](CodeAlpha_DiseasePrediction_from_Medical_data/README.md) for details.

## Quickstart

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r CodeAlpha_Credit_Scoring_Model/requirements.txt
pip install -r CodeAlpha_DiseasePrediction_from_Medical_data/requirements.txt
```

3. Run a demo training session:

```powershell
# Credit scoring example
python CodeAlpha_Credit_Scoring_Model/eda_and_training.py

# Disease prediction example
python CodeAlpha_DiseasePrediction_from_Medical_data/main.py
```

## Repository Structure

- `CodeAlpha_Credit_Scoring_Model/` — Data, EDA and training scripts, dataset `credit_data.csv`.
- `CodeAlpha_DiseasePrediction_from_Medical_data/` — Data loader, preprocessing, model code and entrypoint.

## Highlights

- Clear, modular code organized by pipeline stage (load → preprocess → model → evaluate).
- Reproducible experiments with separated requirements for each project.
- Lightweight and easy to extend for new datasets or model architectures.



