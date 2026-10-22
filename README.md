# Structural Break Detection in Multi-Domain Time Series

Solution to the **ADIA Lab Structural Break Open Benchmark**, hosted on CrunchDAO.

[Competition page](https://hub.crunchdao.com/competitions/structural-break-open-benchmark) · [Author profile](https://hub.crunchdao.com/users/tight-diego)

---

## Overview

Each sample is a univariate time series split into two segments: a *pre* period (`period = 0`) and a *post* period (`period = 1`). The task is to estimate the probability that a structural break occurs at the boundary between them, such as a change in level, volatility, trend or dependence structure. The series come from heterogeneous domains, including financial markets, climate, biometrics and industrial sensors, so the solution has to work without domain-specific assumptions.

- **Training set:** 10,001 series (about 23.7M observations)
- **Class balance:** about 29% positive
- **Metric:** ROC AUC

## Results

| Evaluation                          | ROC AUC    |
| ----------------------------------- | ---------- |
| **Official leaderboard (CrunchDAO)** | **0.7928** |
| 5-fold stratified cross-validation  | 0.7960     |
| Local hold-out (`X_test.reduced`)   | 0.7678     |

**Final ranking: 29th place.**

### Model evolution

Each step was validated with 5-fold stratified cross-validation on the training set.

| Stage                                   | ROC AUC |
| --------------------------------------- | ------: |
| `tsfresh` baseline (pre / post / full)  |  0.6620 |
| + pre/post interactions (diff, ratio)   |  0.7134 |
| + cumulative-sum representations        |  0.7256 |
| Importance-based reduction to Top 200   |  0.7690 |
| + HMM regime features (206 features)    |  0.7713 |
| Soft-voting ensemble                    |  0.7829 |
| Feature pruning to Top 75               |  0.7933 |
| Optuna-tuned ensemble (final)           |  0.7960 |

## Approach

### 1. Feature engineering

Statistical descriptors are extracted with `tsfresh` independently from the pre, post and full segments. The key step is turning static descriptors into **change descriptors**: for each feature available in both segments, the pipeline computes the post–pre difference and the post/pre ratio. It also extracts features from cumulative-sum transforms of the series, both global and per period. These amplify slow drifts and accumulated imbalances that are hard to see pointwise. A global coefficient of variation (`magic_cv_full`) completes the set.

### 2. Regime modelling with Hidden Markov Models

A two-state Gaussian HMM is fitted on the standardized pre segment and then used to score the post segment. A drop in per-step log-likelihood means the post regime is poorly explained by the dynamics learned before the boundary. A second HMM, fitted on the full series, flags whether the Viterbi path switches state exactly at the boundary. These features consistently rank among the most important ones.

### 3. Feature selection

Starting from the 206 strongest candidates (Top 200 + HMM), XGBoost feature importance was used to prune the set to **75 features**. Removing the noisy tail improved every tree-based learner in the ensemble.

### 4. Model

A soft-voting ensemble of three gradient-boosted tree models, all with `scale_pos_weight` set to the class-imbalance ratio:

- **XGBoost**: hyperparameters tuned with Optuna (TPE)
- **LightGBM**: tuned with Optuna, with `num_leaves` constrained by `max_depth`
- **CatBoost**: fixed, robust configuration

### What did not work

- **TimesFM forecasts** of the post segment, used as an "expectation vs. reality" signal: on their own they reached only 0.54 AUC. The foundation model smooths out the micro-structural changes that matter for this task.
- **Econophysics descriptors** (Hurst exponent, permutation entropy) added no signal.
- **Per-cluster models** built on UMAP + HDBSCAN domain clusters did not beat a single global model.

## Pipeline

```text
raw series (id, time, value, period)
        │
        ├── tsfresh on pre / post / full ───────┐
        ├── post–pre differences and ratios ────┤
        ├── tsfresh on cumulative sums ─────────┼──► Top 75 features
        └── HMM regime scanner ─────────────────┘           │
                                                            ▼
                                     XGBoost + LightGBM + CatBoost
                                            (soft voting)
                                                            │
                                                            ▼
                                             P(structural break)
```

## Repository structure

```text
├── data/
│   ├── raw/                              Competition data (not tracked)
│   └── processed/                        Generated feature stores (not tracked)
├── models/                               Serialized models (not tracked)
├── notebooks/
│   ├── 01_EDA_Clustering.ipynb           Exploratory analysis, UMAP / HDBSCAN clustering
│   ├── 02_Feature_Engineering.ipynb      tsfresh, interaction and cumulative-sum feature stores
│   ├── 03_Model_Evolution.ipynb          Incremental benchmark of feature families
│   ├── 04a_TimesFM_Predictions.ipynb     TimesFM forecasts of the post segment
│   ├── 04b_TimesFM_Feature_Engineering.ipynb
│   ├── 04c_TimesFM_Model.ipynb           TimesFM feature evaluation (discarded)
│   ├── 05a_HMM_Feature_Engineering.ipynb Hidden Markov Model regime features
│   ├── 05b_HMM_Model.ipynb               HMM feature evaluation
│   ├── 06_Ensembling.ipynb               XGBoost / LightGBM / CatBoost ensemble
│   ├── 07_Feature_Selection.ipynb        Importance-based pruning to 75 features
│   ├── 08_Hyperparameter_Tuning.ipynb    Optuna search
│   ├── 09_Final_Model.ipynb              End-to-end pipeline and local evaluation
│   └── 10_Submission.ipynb               CrunchDAO submission notebook
├── src/
│   ├── config.py                         Hyperparameters and final feature list
│   └── main.py                           Production train() / infer() pipeline
└── requirements.txt
```

## Reproducing the results

**1. Set up the environment** (Python 3.10+):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the data.** Download the competition files from CrunchDAO and place them in `data/raw/`:

```text
X_train.parquet   y_train.parquet   X_test.reduced.parquet   y_test.reduced.parquet
```

**3. Train and predict.** `src/main.py` exposes the same `train()` / `infer()` interface used by the CrunchDAO runner. Feature extraction is rebuilt from the raw series, so no intermediate files are needed:

```python
import pandas as pd
from main import train, infer   # run from the src/ directory

X_train = pd.read_parquet("../data/raw/X_train.parquet")
y_train = pd.read_parquet("../data/raw/y_train.parquet")
X_test  = pd.read_parquet("../data/raw/X_test.reduced.parquet")

train(X_train, y_train)
predictions = infer(X_test)     # DataFrame of break probabilities indexed by series id
```

The notebooks document the full research process and are meant to be run in order. Each one reads from and writes to `data/processed/`.

## Author

**Diego** · [tight-diego on CrunchDAO](https://hub.crunchdao.com/users/tight-diego)
