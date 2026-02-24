# F1 Circuit Clustering & Race Gap Prediction

This repository contains:

- **ClusteringModel.py**: Generates circuit clusters using K-Means, UMAP, and HDBSCAN based on track features. Outputs circuit_clusters.csv and visualizations.
- **FinalF1Pipeline.qmd**: End-to-end XGBoost race gap prediction pipeline, including feature engineering, temporal cross-validation, hyperparameter tuning, SHAP analysis, and model artifact saving.
- **ClusteredResults.csv**: Main training dataset for the XGBoost model, containing engineered features and target values.
- **abu_dhabi_2025_predict.csv**, **azerbaijan_2025_predict.csv**, **cota_2025_predict.csv**: Validation datasets for inference and model evaluation on unseen races.

## Project Overview
- Circuit clustering is used as a novel preprocessing step to enhance race gap prediction.
- The XGBoost model is trained on 2023–2025 FastF1 data, with temporal splits and robust feature engineering.

## File Descriptions
- **ClusteringModel.py**: Track clustering script.
- **FinalF1Pipeline.qmd**: Full XGBoost pipeline and analysis.
- **ClusteredResults.csv**: Training data.
- **abu_dhabi_2025_predict.csv**, **azerbaijan_2025_predict.csv**, **cota_2025_predict.csv**: Validation data.

For details, see comments in each file.
