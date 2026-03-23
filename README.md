# ADIA Lab: Structural Break Open Benchmark Challenge

This repository contains the complete solution for the ADIA Lab Structural Break Open Benchmark Challenge. The objective is to detect structural breaks in multi-domain time series, encompassing financial markets, climate data, biometrics, and industrial sensors.

**🏆 Final Performance:** * **5-Fold Cross-Validation (ROC AUC):** 0.7967
* **Out-of-Sample Test (ROC AUC):** 0.7678

## 🧠 Architecture and Methodology

The project was developed through an exhaustive experimental process, fully documented in the `notebooks/` directory:

1. **Feature Engineering (`tsfresh`):** Massive extraction of statistical time-series features, subsequently reduced to a Top 200 baseline.
2. **Sequential Models & Domain Intelligence:**
   - Pure Deep Learning architectures (like TimesFM) were evaluated but ultimately discarded due to the excessive smoothing of micro-structural signals.
   - **Hidden Markov Models (HMM)** were successfully implemented to capture differences in topological probability (log-likelihood) between pre- and post-break regimes.
3. **Feature Selection (Noise Ablation):** We discovered that the original dataset contained significant noise that penalized tree-based algorithms. Using XGBoost Feature Importance, we reduced the dataset from 206 features to the **Top 75 most predictive variables**.
4. **Ensembling & Hyperparameter Tuning (Optuna):** The final model is a *Soft Voting Classifier* consisting of:
   - **XGBoost** (Tuned via Optuna)
   - **LightGBM** (Tuned via Optuna, specifically restricting the `num_leaves` asymmetry)
   - **CatBoost** (Robust default configuration)

## 📁 Project Structure

```text
├── data/
│   ├── processed/     # Extracted and cleaned features (.parquet) [Git Ignored]
│   └── raw/           # Original ADIA Lab dataset (.parquet) [Git Ignored]
├── models/            # Final serialized models [Git Ignored]
├── notebooks/         # Experimentation logs (01 to 11)
├── src/               # Production source code (config.py, main.py)
├── .gitignore         # Exclusion rules for GitHub
├── README.md          # Project documentation
└── requirements.txt   # Dependencies to reproduce the environment