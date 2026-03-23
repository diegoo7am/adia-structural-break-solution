"""
Configuration file for the Structural Break Detection model.
Contains hyperparameters, paths, and feature lists.
"""
from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# --- BEST HYPERPARAMETERS (OPTUNA TOP 75) ---
BEST_XGB_PARAMS = {
    'n_estimators': 350, 
    'learning_rate': 0.05083158501542684, 
    'max_depth': 8, 
    'subsample': 0.9729161367647149, 
    'colsample_bytree': 0.6061470949312417, 
    'min_child_weight': 5, 
    'gamma': 0.007572118551378439, 
    'eval_metric': 'auc', 
    'random_state': 42, 
    'n_jobs': -1
}

BEST_LGB_PARAMS = {
    'max_depth': 8, 
    'n_estimators': 350, 
    'learning_rate': 0.03163661347608696, 
    'num_leaves': 169, 
    'min_child_samples': 58, 
    'subsample': 0.9156545808727808, 
    'colsample_bytree': 0.7755355038366832, 
    'random_state': 42, 
    'n_jobs': -1, 
    'verbose': -1
}

# --- FEATURE LIST ---
TOP_75_FEATURES = [
    "hmm_post_ll_per_step",
    "value__quantile__q_0.2_full",
    "cumsum_period__agg_autocorrelation__f_agg_\"mean\"__maxlag_40_cumsum_pre",
    "value__percentage_of_reoccurring_values_to_all_values_ratio",
    "magic_cv_full",
    "value__ratio_value_number_to_time_series_length_diff",
    "value__has_duplicate_full",
    "value__quantile__q_0.4_full",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_0.8__ql_0.4_full",
    "hmm_ll_ratio",
    "cumsum_full__fft_coefficient__attr_\"abs\"__coeff_1_cumsum_full",
    "cumsum_period__fft_coefficient__attr_\"abs\"__coeff_1_cumsum_pre",
    "value__change_quantiles__f_agg_\"var\"__isabs_False__qh_1.0__ql_0.2_ratio",
    "cumsum_full__first_location_of_maximum_cumsum_full",
    "value__agg_linear_trend__attr_\"intercept\"__chunk_len_5__f_agg_\"min\"_pre",
    "value__percentage_of_reoccurring_values_to_all_values_diff",
    "cumsum_full__cid_ce__normalize_False_cumsum_full",
    "cumsum_period__fft_coefficient__attr_\"abs\"__coeff_0_cumsum_pre",
    "value__variation_coefficient_full",
    "cumsum_full__augmented_dickey_fuller__attr_\"teststat\"__autolag_\"AIC\"_cumsum_full",
    "value__agg_linear_trend__attr_\"rvalue\"__chunk_len_10__f_agg_\"mean\"_diff",
    "value__change_quantiles__f_agg_\"var\"__isabs_True__qh_0.6__ql_0.4_ratio",
    "value__index_mass_quantile__q_0.1_post",
    "cumsum_period__agg_autocorrelation__f_agg_\"median\"__maxlag_40_cumsum_pre",
    "value__variance_ratio",
    "cumsum_full__agg_autocorrelation__f_agg_\"median\"__maxlag_40_cumsum_full",
    "value__index_mass_quantile__q_0.9_pre",
    "value__agg_linear_trend__attr_\"rvalue\"__chunk_len_5__f_agg_\"mean\"_full",
    "value__change_quantiles__f_agg_\"var\"__isabs_False__qh_0.6__ql_0.2_full",
    "cumsum_full__fft_coefficient__attr_\"abs\"__coeff_0_cumsum_full",
    "value__quantile__q_0.7_full",
    "value__quantile__q_0.3_post",
    "hmm_ll_diff",
    "cumsum_full__index_mass_quantile__q_0.6_cumsum_full",
    "value__percentage_of_reoccurring_values_to_all_values_pre",
    "value__index_mass_quantile__q_0.4_post",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_1.0__ql_0.0_ratio",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_0.6__ql_0.0_full",
    "value__change_quantiles__f_agg_\"var\"__isabs_True__qh_0.8__ql_0.0_full",
    "cumsum_period__change_quantiles__f_agg_\"mean\"__isabs_False__qh_1.0__ql_0.6_cumsum_post",
    "cumsum_full__change_quantiles__f_agg_\"mean\"__isabs_False__qh_1.0__ql_0.8_cumsum_full",
    "value__percentage_of_reoccurring_datapoints_to_all_datapoints_full",
    "cumsum_period__absolute_maximum_cumsum_pre",
    "value__root_mean_square_ratio",
    "value__index_mass_quantile__q_0.3_post",
    "value__agg_linear_trend__attr_\"rvalue\"__chunk_len_10__f_agg_\"mean\"_full",
    "cumsum_full__cid_ce__normalize_True_cumsum_full",
    "value__ratio_beyond_r_sigma__r_1_pre",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_1.0__ql_0.8_ratio",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_1.0__ql_0.2_ratio",
    "value__median_ratio",
    "cumsum_period__change_quantiles__f_agg_\"var\"__isabs_True__qh_0.8__ql_0.0_cumsum_pre",
    "value__change_quantiles__f_agg_\"var\"__isabs_True__qh_1.0__ql_0.2_ratio",
    "value__quantile__q_0.4_ratio",
    "value__kurtosis_full",
    "value__ar_coefficient__coeff_0__k_10_ratio",
    "cumsum_full__change_quantiles__f_agg_\"mean\"__isabs_True__qh_0.6__ql_0.2_cumsum_full",
    "value__percentage_of_reoccurring_datapoints_to_all_datapoints_diff",
    "hmm_viterbi_transition",
    "value__agg_linear_trend__attr_\"intercept\"__chunk_len_50__f_agg_\"mean\"_full",
    "value__ratio_beyond_r_sigma__r_1.5_full",
    "cumsum_period__change_quantiles__f_agg_\"mean\"__isabs_False__qh_0.4__ql_0.0_cumsum_pre",
    "value__change_quantiles__f_agg_\"var\"__isabs_False__qh_0.8__ql_0.4_ratio",
    "cumsum_full__fft_aggregated__aggtype_\"kurtosis\"_cumsum_full",
    "cumsum_period__change_quantiles__f_agg_\"mean\"__isabs_True__qh_0.6__ql_0.0_cumsum_pre",
    "value__change_quantiles__f_agg_\"var\"__isabs_False__qh_0.8__ql_0.2_post",
    "value__ratio_beyond_r_sigma__r_0.5_pre",
    "value__agg_linear_trend__attr_\"intercept\"__chunk_len_50__f_agg_\"var\"_ratio",
    "value__change_quantiles__f_agg_\"mean\"__isabs_True__qh_0.8__ql_0.4_ratio",
    "value__quantile__q_0.9_ratio",
    "value__ratio_beyond_r_sigma__r_3_pre",
    "value__variance_diff",
    "value__quantile__q_0.1_ratio",
    "cumsum_full__autocorrelation__lag_1_cumsum_full",
    "cumsum_period__change_quantiles__f_agg_\"var\"__isabs_True__qh_0.6__ql_0.0_cumsum_post"
]