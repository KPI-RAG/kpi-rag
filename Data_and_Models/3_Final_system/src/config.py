"""Single source of truth for paths, seeds and model settings.

Set the environment variable KPI_RAG_ROOT to run on another machine;
it defaults to the project folder on the Queen's JupyterHub server.
"""
import os

ROOT     = os.environ.get("KPI_RAG_ROOT", "/home/jovyan/kpi_rag")
DATA_DIR = os.path.join(ROOT, "Models Improvement", "Data Splits")
# Cache of the REAL dataset parsed once by Synthetic_Data_Part1.ipynb (cell 2). It only
# lives in the "Synthetic" folder; no synthetic windows are used anywhere in Phase 2.
REAL_PARSED_DIR = os.path.join(ROOT, "Models Improvement", "Synthetic")
CSV_PATH = os.path.join(ROOT, "data", "telecomts.csv")
PHASE2   = os.path.join(ROOT, "phase2")
RESULTS  = os.path.join(PHASE2, "results")

# Canonical split files (inputs, read-only)
SPLIT_FILES = {
    "train_X":     "train_X_sessionsplit_fixed.npy",   # (25688, 64) Config A features
    "train_y":     "train_y_sessionsplit.npy",         # string labels
    "train_idx":   "train_idx_sessionsplit.npy",       # CSV row index of each train row
    "test_idx":    "test_idx_sessionsplit.npy",        # CSV row index of each test row
    "session_ids": "session_id_by_row.npy",            # session id of every CSV row
}
RAW_KPIS = os.path.join(REAL_PARSED_DIR, "raw_kpis.npy")   # (32000, 16, 128) real signals
RAW_META = os.path.join(REAL_PARSED_DIR, "raw_meta.csv")   # real label per CSV row

NORMAL_LABEL   = "Normal"
CATCHALL_LABEL = "Co-Channel Interference (Mild)"
N_FEATURES_A   = 64        # Config A = mean/std/min/max of 16 KPI channels

# Evaluation protocol
SEED       = 42
CV_SEEDS   = [42, 7, 123, 2024, 99]
N_FOLDS    = 5
THREADS    = 4

# Phase 1 baseline (final_confirm_records.csv) - must reproduce
BASELINE_MEAN = 0.2908
BASELINE_STD  = 0.0902
BASELINE_TOL  = 0.005

# Phase 1 final hyper-parameters (RandomizedSearchCV, submitted notebook [8]).
# Phase 1's make_base_clf() created every CatBoost model with auto_class_weights="Balanced";
# for Model 1 this is applied ON TOP of the 80/20 rebalancing.
MODEL1_PARAMS = dict(learning_rate=0.05, l2_leaf_reg=5, iterations=200, depth=4,
                     auto_class_weights="Balanced")
MODEL2_PARAMS = dict(learning_rate=0.2, l2_leaf_reg=7, iterations=200, depth=4,
                     auto_class_weights="Balanced")

# ADOPTED Model 1 (decision 2026-10-13, after step 03): correctly parsed features +
# imbalance corrected once (80/20 rebalance only, no "Balanced" weights). Threshold 0.5.
MODEL1_FIXED_FEATURES  = True
MODEL1_CLASS_WEIGHTING = False
MODEL1_THRESHOLD       = 0.5

# Phase 1 Model 1 test confusion matrix at threshold 0.5 (sanity reference)
PHASE1_TEST_CM = {"TN": 4907, "FP": 1105, "FN": 31, "TP": 269}

# Task 2 pre-registered threshold rule
MIN_RECALL = 0.85
