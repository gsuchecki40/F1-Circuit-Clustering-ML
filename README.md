# F1 Race Outcome Prediction | XGBoost + Circuit Clustering

XGBoost regression model for predicting Formula 1 race finishing gaps, trained on FastF1 data from the 2023–2025 seasons. The central research question driving this project: **can unsupervised circuit clustering as a preprocessing step improve the prediction of Formula 1 race outcomes using publicly available data?**

This is an ongoing project. The current iteration focuses on incorporating circuit characteristics into the prediction pipeline. Future work includes live strategy modeling, safety car prediction, and qualifying result forecasting.

---

## Research Paper

A full writeup of the methodology, clustering approach, SHAP analysis, and validation results is included in `F1_Clustering_Report.pdf`. If you want the reasoning behind design decisions or a deeper look at what the model is actually doing, start there.

---

## How It Works

Rather than treating every circuit the same, the model first groups all 24 circuits on the calendar into 6 distinct track types using K-Means clustering. Each cluster is defined by layout and racing characteristics — corner profiles, average speed, tyre degradation, safety car probability, altitude, and more. That cluster assignment then feeds into the XGBoost model as a feature alongside driver and race data.

The model predicts the **gap to the race winner** (log-transformed, min-max scaled within each race) rather than a discrete finishing position. Positions are derived by sorting predicted gaps. This continuous target captures relative competitiveness across the grid rather than collapsing everything into a ranked list.

---

## Circuit Clusters (K-Means, k=6)

| Cluster | Label | Representative Circuits |
|---------|-------|------------------------|
| C0 | Flat & Open | Montreal, Qatar |
| C1 | Narrow Street | Monaco, Singapore |
| C2 | Long Straight, Low Stress | Baku, Abu Dhabi, Miami, Las Vegas, Shanghai, Melbourne, Imola |
| C3 | Power Unit | Monza |
| C4 | Technical Altitude | Bahrain, COTA, Hungary, Barcelona, Zandvoort, Brazil, Mexico, Red Bull Ring |
| C5 | High-Speed Corner Load | Spa, Silverstone, Suzuka, Jeddah |

---

## Key Features (~70 per driver per race)

- Grid position and qualifying gap to pole
- 5-race rolling driver and team form (points and finish position)
- Circuit cluster assignment (K-Means)
- Circuit characteristics: track length, corner counts, full throttle %, average speed, longest straight, DRS zones, altitude, tyre degradation, pit lane time loss, safety car probability, overtaking difficulty
- Weather: air temp, track temp, humidity, wind speed, wind direction, pressure

---

## Top SHAP Features

Based on GradientBoostingRegressor SHAP analysis across 2023–2025 seasons:

1. Grid Position
2. Driver rolling avg points (5 races)
3. Driver rolling avg finish (5 races)
4. Team rolling avg points (5 races)
5. Track temperature
6. Pit lane time loss
7. Altitude
8. Average speed
9. Air temperature
10. Qualifying gap (scaled)

---

## Validation Results

The model was trained on 2023–2024 data and validated on unseen 2025 races using temporal cross-validation (no data leakage).

| Race | MAE | Exact | Within ±3 |
|------|-----|-------|-----------|
| Azerbaijan GP (Round 17) | 2.80 | 2/20 | 14/20 |
| United States GP (Round 19) | 4.60 | 3/20 | 8/20 |

Baku: correctly predicted Verstappen's victory. Primary error source was Piastri's Lap 1 DNF and the resulting safety car compressing the field.

COTA: podium predicted exactly (Verstappen, Norris, Leclerc). Variance driven by Sainz DNF and cascade contact with Antonelli, plus Sauber significantly outperforming rolling form in extreme heat.

---

## Repo Structure

```
├── ClusteringModel.py              # K-Means + HDBSCAN circuit clustering, PCA + UMAP visualization
├── merge_clusters.py               # Merges cluster labels into main race results dataset
├── FinalF1Pipeline.qmd             # Full XGBoost pipeline — feature engineering, model training, SHAP analysis
├── FinalMergedWithCircuits.csv     # Base dataset — race results merged with raw circuit characteristics
├── ClusteredResults.csv            # Race results (2023–2025) with circuit characteristics and cluster labels
├── azerbaijan_2025_predict.csv     # Model input for 2025 Azerbaijan GP prediction
├── cota_2025_predict.csv           # Model input for 2025 United States GP prediction
├── abu_dhabi_2025_predict.csv      # Model input for 2025 Abu Dhabi GP prediction
└── docs/                           # Research paper and presentation slides
```

---

## Dependencies

```
xgboost
scikit-learn
pandas
numpy
shap
umap-learn
hdbscan
fastf1
matplotlib
seaborn
plotnine
```

---

## Usage

### Step 1 — Run the Clustering Model

`ClusteringModel.py` expects a file called `FinalMergedWithCircuits.csv` in the working directory — this is already included in the repo and will be updated after the first few races of the 2026 season. It aggregates circuit features by median, normalizes them, runs PCA, and outputs two cluster label columns: `kmeans` (6 clusters) and `hdbscan`.

```bash
python ClusteringModel.py
```

Outputs:
- `circuit_clusters.csv` — one row per circuit with cluster labels and UMAP coordinates
- `circuit_clusters_umap.png` — UMAP visualization of the clusters

> Note: HDBSCAN assigned most circuits to cluster -1 (noise) due to the small dataset size. The pipeline uses `kmeans` only — `hdbscan` is dropped before model training.

---

### Step 2 — Merge Cluster Labels into the Race Results

`merge_clusters.py` joins the cluster labels from `circuit_clusters.csv` back onto `FinalMergedWithCircuits.csv` (already included in the repo), matching on `circuit_id` and `circuit_name`.

```bash
python merge_clusters.py
```

Outputs:
- `FinalMergedWithCircuits_with_clusters.csv` — full race results dataset with `kmeans` and `hdbscan` columns appended

Rename this file to `ClusteredResults.csv` (or update the path in the pipeline) before running Step 3.

---

### Step 3 — Run the Prediction Pipeline

`FinalF1Pipeline.qmd` is a Quarto notebook that handles everything from feature engineering to model training and evaluation. It expects `ClusteredResults.csv` in the working directory.

```bash
quarto render FinalF1Pipeline.qmd
```

Or run interactively cell-by-cell in VS Code with the Quarto extension.

**What the pipeline does in order:**
1. Loads `ClusteredResults.csv` and computes the log-transformed, race-scaled gap-to-winner target
2. Builds qualifying gap features (scaled relative to pole time within each race)
3. Computes rolling 5-race driver and team form features with `shift(1)` to prevent data leakage
4. One-hot encodes `circuit_name` and `track_type`
5. Drops target-leaking columns and rows with missing targets
6. Splits temporally — trains on 2023–2024, tests on 2025
7. Runs sequential hyperparameter tuning via temporal cross-validation (depth, gamma, subsample, regularization, learning rate)
8. Trains the final model and evaluates on the 2025 test set
9. Generates SHAP summary and waterfall plots, and a feature importance chart

Outputs: `shap_summary.png` and rendered HTML report.

---

### Making a Prediction for a New Race

To generate predictions for an upcoming race, build an input CSV matching the structure of the prediction CSVs included in the repo (`azerbaijan_2025_predict.csv`, `cota_2025_predict.csv`, `abu_dhabi_2025_predict.csv`). Each row is one driver with their grid position, qualifying time, weather conditions, and circuit characteristics for that round. The `kmeans` cluster for the circuit must be looked up from `circuit_clusters.csv` and included as a column.

Feed the CSV into the trained model by loading it as a DMatrix and calling `model_final.predict()`. Sort predicted gaps ascending to get the finishing order.

---

## Limitations & Future Work

- Model trained through mid-2025. 2026 regulation reset introduces significant uncertainty — rolling form features zeroed for season opener.
- Safety car and incident outcomes are the primary source of prediction error and are not currently modeled.
- Planned additions: safety car probability as a live feature, qualifying prediction module, active aero zones to replace DRS under 2026 regs, full strategy simulation.
