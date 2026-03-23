"""
Main execution script for Structural Break Detection.
Provides train() and infer() functions for production environments.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from tsfresh import extract_features
from tsfresh.feature_extraction.settings import from_columns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import re
import warnings

# Import configuration from config.py
from config import BEST_XGB_PARAMS, BEST_LGB_PARAMS, TOP_75_FEATURES

warnings.filterwarnings('ignore')

# Global model state
GLOBAL_MODEL = None

def build_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """Extracts features dynamically based on TOP_75_FEATURES suffix parsing."""
    print("\n>>> Building features pipeline initialized...")

    if isinstance(X_raw.index, pd.MultiIndex) or 'id' not in X_raw.columns:
        X_raw = X_raw.reset_index()
        
    unique_ids = X_raw['id'].unique()
    print(f"    -> Total unique time series detected: {len(unique_ids)}")
    
    # --- 1. PREPARE SUB-DATASETS ---
    print("--- [Step 1/6] Preparing sub-datasets (pre, post, cumsum)...")
    df_pre = X_raw[X_raw['period'] == 0].copy()
    df_post = X_raw[X_raw['period'] == 1].copy()
    
    df_cs_full = X_raw.copy()
    df_cs_full['cumsum_full'] = df_cs_full.groupby('id')['value'].cumsum()
    
    df_cs_period = X_raw.copy()
    df_cs_period['cumsum_period'] = df_cs_period.groupby(['id', 'period'])['value'].cumsum()
    df_cs_pre = df_cs_period[df_cs_period['period'] == 0]
    df_cs_post = df_cs_period[df_cs_period['period'] == 1]

    # --- 2. PARSE REQUIRED TSFRESH FEATURES ---
    print("--- [Step 2/6] Parsing required tsfresh features from TOP_75_FEATURES...")
    ts_full, ts_pre, ts_post = set(), set(), set()
    ts_cs_full, ts_cs_pre, ts_cs_post = set(), set(), set()
    
    for f in TOP_75_FEATURES:
        if f.startswith('hmm_'): 
            continue # HMM features are handled separately
            
        if f == 'magic_cv_full':
            ts_full.update(['value__variance', 'value__mean'])
        elif f.endswith('_ratio'):
            base_f = f[:-6]
            ts_pre.add(base_f)
            ts_post.add(base_f)
        elif f.endswith('_diff'):
            base_f = f[:-5]
            ts_pre.add(base_f)
            ts_post.add(base_f)
        elif f.endswith('_cumsum_full'): 
            ts_cs_full.add(f[:-12])
        elif f.endswith('_cumsum_pre'): 
            ts_cs_pre.add(f[:-11])
        elif f.endswith('_cumsum_post'): 
            ts_cs_post.add(f[:-12])
        elif f.endswith('_full'): 
            ts_full.add(f[:-5])
        elif f.endswith('_pre'): 
            ts_pre.add(f[:-4])
        elif f.endswith('_post'): 
            ts_post.add(f[:-5])

    # --- 3. EXTRACT TARGETED TSFRESH FEATURES ---
    print("--- [Step 3/6] Extracting targeted tsfresh features...")
    df_final = pd.DataFrame(index=unique_ids)
    
    if ts_full:
        print(f"    -> Extracting {len(ts_full)} features from the 'full' segment...")
        feat = extract_features(X_raw, column_id='id', column_sort='time', column_value='value', kind_to_fc_parameters=from_columns(list(ts_full)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_full'))
        
    if ts_pre:
        print(f"    -> Extracting {len(ts_pre)} features from the 'pre' segment...")
        feat = extract_features(df_pre, column_id='id', column_sort='time', column_value='value', kind_to_fc_parameters=from_columns(list(ts_pre)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_pre'))
        
    if ts_post:
        print(f"    -> Extracting {len(ts_post)} features from the 'post' segment...")
        feat = extract_features(df_post, column_id='id', column_sort='time', column_value='value', kind_to_fc_parameters=from_columns(list(ts_post)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_post'))
        
    if ts_cs_full:
        print(f"    -> Extracting {len(ts_cs_full)} features from the 'cumsum_full' segment...")
        feat = extract_features(df_cs_full, column_id='id', column_sort='time', column_value='cumsum_full', kind_to_fc_parameters=from_columns(list(ts_cs_full)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_cumsum_full'))
        
    if ts_cs_pre:
        print(f"    -> Extracting {len(ts_cs_pre)} features from the 'cumsum_pre' segment...")
        feat = extract_features(df_cs_pre, column_id='id', column_sort='time', column_value='cumsum_period', kind_to_fc_parameters=from_columns(list(ts_cs_pre)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_cumsum_pre'))
        
    if ts_cs_post:
        print(f"    -> Extracting {len(ts_cs_post)} features from the 'cumsum_post' segment...")
        feat = extract_features(df_cs_post, column_id='id', column_sort='time', column_value='cumsum_period', kind_to_fc_parameters=from_columns(list(ts_cs_post)), disable_progressbar=True)
        df_final = df_final.join(feat.add_suffix('_cumsum_post'))

    # --- 4. CALCULATE CUSTOM FEATURES ---
    print("--- [Step 4/6] Calculating custom interaction features (ratios, diffs, Magic CV)...")
    if 'value__variance_full' in df_final.columns and 'value__mean_full' in df_final.columns:
        std_full = np.sqrt(df_final['value__variance_full'])
        df_final['magic_cv_full'] = std_full / (df_final['value__mean_full'] + 1e-8)
        
    for f in TOP_75_FEATURES:
        if f.endswith('_ratio'):
            base = f[:-6]
            if f"{base}_pre" in df_final.columns and f"{base}_post" in df_final.columns:
                df_final[f] = df_final[f"{base}_post"] / (df_final[f"{base}_pre"] + 1e-8)
        elif f.endswith('_diff'):
            base = f[:-5]
            if f"{base}_pre" in df_final.columns and f"{base}_post" in df_final.columns:
                df_final[f] = df_final[f"{base}_post"] - df_final[f"{base}_pre"]

    # --- 5. HMM REGIME SCANNER ---
    hmm_needed = any(f.startswith('hmm_') for f in TOP_75_FEATURES)
    if hmm_needed:
        print("--- [Step 5/6] Initializing Hidden Markov Model Regime Scanner...")
        hmm_features = []
        for uid in unique_ids:
            series_data = X_raw[X_raw['id'] == uid].sort_values('time')
            pre_vals = series_data[series_data['period'] == 0]['value'].values.reshape(-1, 1)
            post_vals = series_data[series_data['period'] == 1]['value'].values.reshape(-1, 1)
            all_vals = series_data['value'].values.reshape(-1, 1)
            
            feat_dict = {'id': uid, 'hmm_ll_diff': 0, 'hmm_ll_ratio': 0, 'hmm_post_ll_per_step': 0, 'hmm_viterbi_transition': 0}
            
            if len(pre_vals) >= 20 and len(post_vals) >= 5:
                try:
                    scaler = StandardScaler()
                    pre_scaled = scaler.fit_transform(pre_vals)
                    post_scaled = scaler.transform(post_vals)
                    
                    model_A = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=42)
                    model_A.fit(pre_scaled)
                    
                    norm_ll_pre = model_A.score(pre_scaled) / len(pre_scaled)
                    norm_ll_post = model_A.score(post_scaled) / len(post_scaled)
                    
                    feat_dict['hmm_ll_diff'] = norm_ll_pre - norm_ll_post
                    feat_dict['hmm_ll_ratio'] = abs(norm_ll_post) / (abs(norm_ll_pre) + 1e-6)
                    feat_dict['hmm_post_ll_per_step'] = norm_ll_post
                    
                    scaler_all = StandardScaler()
                    all_scaled = scaler_all.fit_transform(all_vals)
                    model_B = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=42)
                    model_B.fit(all_scaled)
                    hidden_states = model_B.predict(all_scaled)
                    
                    states_pre = hidden_states[:len(pre_scaled)]
                    states_post = hidden_states[len(pre_scaled):]
                    if states_pre[-1] != states_post[0]:
                        feat_dict['hmm_viterbi_transition'] = 1
                except Exception:
                    pass
            hmm_features.append(feat_dict)
            
        df_hmm = pd.DataFrame(hmm_features).set_index('id')
        df_final = df_final.join(df_hmm)
        print(f"    -> HMM extraction completed for {len(unique_ids)} series.")
    else:
        print("--- [Step 5/6] HMM Regime Scanner skipped (No HMM features requested).")

    # --- 6. CLEANUP & FINAL ALIGNMENT ---
    print("--- [Step 6/6] Cleaning up DataFrames and aligning final columns...")
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in TOP_75_FEATURES:
        if col not in df_final.columns:
            print(f"    -> WARNING: Feature '{col}' not generated. Imputing with 0.")
            df_final[col] = 0

    df_final = df_final[TOP_75_FEATURES]

    df_final.columns = df_final.columns.str.replace(r'[^\w\s]', '_', regex=True)

    print(">>> Feature engineering pipeline completed successfully!\n")
    return df_final


def train(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Trains the Voting Ensemble Model."""
    global GLOBAL_MODEL
    print("--- STARTING TRAINING PHASE ---")
    
    X_features = build_features(X_train)
    y = y_train.loc[X_features.index]['structural_breakpoint']
    imbalance_ratio = float((y == 0).sum() / (y == 1).sum())
    
    print(f"-> Class imbalance ratio calculated: {imbalance_ratio:.4f}")
    print("-> Fitting XGBoost, LightGBM, and CatBoost models...")

    xgb_model = xgb.XGBClassifier(**BEST_XGB_PARAMS, scale_pos_weight=imbalance_ratio)
    lgb_model = lgb.LGBMClassifier(**BEST_LGB_PARAMS, scale_pos_weight=imbalance_ratio)
    cat_model = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, scale_pos_weight=imbalance_ratio, eval_metric='AUC', random_state=42, verbose=0, thread_count=-1)
    
    GLOBAL_MODEL = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)],
        voting='soft', n_jobs=1
    )
    print("-> Aggregating models into a Soft Voting Classifier...")
    GLOBAL_MODEL.fit(X_features, y)
    print("--- TRAINING COMPLETE ---")


def infer(X_test: pd.DataFrame) -> pd.DataFrame:
    """Generates predictions for new data."""
    global GLOBAL_MODEL
    if GLOBAL_MODEL is None:
        raise ValueError("Model is not trained. Call train() first.")
        
    print("--- STARTING INFERENCE PHASE ---")
    X_features = build_features(X_test)

    print("-> Generating probabilities using the trained Voting Classifier...")
    probabilities = GLOBAL_MODEL.predict_proba(X_features)[:, 1]
    
    print("--- INFERENCE COMPLETE ---")
    return pd.DataFrame(probabilities, index=X_features.index, columns=['structural_breakpoint'])