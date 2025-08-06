#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4. Model Training and Interpretation
-------------------------------------
Modules:
1. Train tree-based models with RandomizedSearchCV (RF, LightGBM, XGBoost, CatBoost)
2. Train TabPFN with 5-fold CV
3. Generate SHAP interpretation artefacts

Outputs:
- Model joblib files
- CV metrics CSVs
- SHAP beeswarm plots, NPY, CSV

Ensure all data paths exist before running.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from joblib import parallel_backend
from tabpfn import TabPFNRegressor

# -----------------------------------------------------------------------------
# Global Config
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = Path("/home/lp43319/projects/Brainstorm/regre/regression/0619_final")
DATA_PATH = BASE_DIR / "data/all_years_regression_data_renamed.csv"
MODEL_DIR = BASE_DIR / "model"
TABPFN_MODEL_DIR = BASE_DIR / "tabpfn_models"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
TABPFN_MODEL_DIR.mkdir(exist_ok=True, parents=True)

TARGETS = [
    "HealthAccessibility", "FoodAccessibility",
    "EduAccessibility", "EntertainmentAccessibility", "Accessibility"
]
PHASES = ["pre", "during", "post"]
PHASE_COL = "PandemicPhase"
DROP_COLS = TARGETS + ["CBG_ID", PHASE_COL, "Year"]

CV = KFold(n_splits=5, shuffle=True, random_state=42)
SCORER = make_scorer(r2_score)

# -----------------------------------------------------------------------------
# 1. Tree Models with Hyperopt
# -----------------------------------------------------------------------------
MODELS = {
    "RF": RandomForestRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, verbosity=0),
    "CatBoost": CatBoostRegressor(loss_function="RMSE", random_state=42, verbose=False, task_type="CPU")
}

PARAM_SPACES = {
    "RF": {
        "model__n_estimators": [200, 400, 600, 800, 1000, 1200, 1500],
        "model__max_depth": [None, 10, 20, 30, 40, 50, 60],
        "model__min_samples_split": [2, 4, 6, 8, 10],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", "log2", 0.4, 0.6, 0.8, 1.0],
        "model__bootstrap": [True, False]
    },
    "LightGBM": {
        "model__n_estimators": [200, 400, 600, 800, 1000, 1200, 1500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 6, 9, 12],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0]
    },
    "XGBoost": {
        "model__n_estimators": [200, 400, 600, 800, 1000, 1200, 1500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 6, 9, 12],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0]
    },
    "CatBoost": {
        "model__iterations": [200, 400, 600, 800, 1000, 1200, 1500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__depth": [4, 6, 8, 10],
        "model__l2_leaf_reg": [1, 3, 5, 7, 9],
        "model__subsample": [0.6, 0.8, 1.0]
    }
}

def train_tree_models():
    csv_out = RESULT_DIR / "tree_models_results.csv"
    if csv_out.exists():
        master = pd.read_csv(csv_out)
    else:
        master = pd.DataFrame(columns=["Model", "Phase", "Target", "R2_CV", "MSE_CV", "MAE_CV"])
    _done = set(zip(master.Model, master.Phase, master.Target))

    df_all = pd.read_csv(DATA_PATH)

    for mdl_name, base_model in MODELS.items():
        for phase in PHASES:
            df_p = df_all[df_all[PHASE_COL] == phase]
            for tgt in TARGETS:
                tag = (mdl_name, phase, tgt)
                if tag in _done:
                    continue

                y = df_p[tgt]
                X = df_p.drop(columns=DROP_COLS, errors="ignore")
                m = y.notna()
                X, y = X[m], y[m]

                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("model", base_model)])
                rsearch = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=PARAM_SPACES[mdl_name],
                    n_iter=100,
                    scoring=SCORER,
                    n_jobs=-1,
                    cv=CV,
                    random_state=42,
                    verbose=0,
                    refit=True
                )
                with parallel_backend("loky"):
                    rsearch.fit(X, y)

                best_pipe = rsearch.best_estimator_
                joblib.dump(best_pipe, MODEL_DIR / f"{mdl_name.lower()}_{phase}_{tgt}.joblib")

                preds = np.zeros_like(y, dtype=float)
                for tr, te in CV.split(X, y):
                    mdl_clone = deepcopy(best_pipe)
                    mdl_clone.fit(X.iloc[tr], y.iloc[tr])
                    preds[te] = mdl_clone.predict(X.iloc[te])

                master.loc[len(master)] = [
                    mdl_name, phase, tgt,
                    r2_score(y, preds),
                    mean_squared_error(y, preds),
                    mean_absolute_error(y, preds)
                ]
                _done.add(tag)

    master.sort_values(["Phase", "Target", "Model"]).to_csv(csv_out, index=False)

# -----------------------------------------------------------------------------
# 2. TabPFN CV
# -----------------------------------------------------------------------------
def encode_impute(df: pd.DataFrame) -> np.ndarray:
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype("category").cat.codes
    df = df.fillna(df.median(numeric_only=True))
    return df.astype("float32").values

def train_tabpfn():
    csv_out = RESULT_DIR / "tabpfn_results.csv"
    if csv_out.exists():
        master = pd.read_csv(csv_out)
    else:
        master = pd.DataFrame(columns=["Model", "Phase", "Target", "R2_CV", "MSE_CV", "MAE_CV"])
    _done = set(zip(master.Model, master.Phase, master.Target))

    df_all = pd.read_csv(DATA_PATH)

    for phase in PHASES:
        df_p = df_all[df_all[PHASE_COL] == phase].copy()
        X_all = encode_impute(df_p.drop(columns=DROP_COLS, errors="ignore"))
        for tgt in TARGETS:
            tag = ("TabPFN", phase, tgt)
            if tag in _done:
                continue

            y_all = df_p[tgt].astype("float32")
            mask = y_all.notna().values
            if mask.sum() < 60:
                continue

            X, y = X_all[mask], y_all.values[mask]
            y_pred_full = np.zeros_like(y)
            fold = 0
            for tr, te in CV.split(X, y):
                fold += 1
                mdl_path = TABPFN_MODEL_DIR / f"tabpfn_{phase}_{tgt}_fold{fold}.joblib"
                if mdl_path.exists():
                    reg = joblib.load(mdl_path)
                else:
                    reg = TabPFNRegressor(device="cuda")
                    reg.fit(X[tr], y[tr])
                    joblib.dump(reg, mdl_path)
                y_pred_full[te] = reg.predict(X[te])

            master.loc[len(master)] = [
                "TabPFN", phase, tgt,
                r2_score(y, y_pred_full),
                mean_squared_error(y, y_pred_full),
                mean_absolute_error(y, y_pred_full)
            ]
            _done.add(tag)

    master.sort_values(["Phase", "Target", "Model"]).to_csv(csv_out, index=False)

# -----------------------------------------------------------------------------
# 3. SHAP Interpretation
# -----------------------------------------------------------------------------
def generate_shap():
    OUT_DIR = BASE_DIR / "shap_all_models"
    PNG_DIR, NPY_DIR, CSV_DIR = [OUT_DIR / sub for sub in ("png", "npy", "csv")]
    for d in (PNG_DIR, NPY_DIR, CSV_DIR):
        d.mkdir(parents=True, exist_ok=True)

    TOP_K = 15
    N_SAMPLE = 2000
    RNG = np.random.default_rng(42)

    df_all = pd.read_csv(DATA_PATH, low_memory=False).replace([np.inf, -np.inf], np.nan)
    sample_idx = {}
    for phase in PHASES:
        for tgt in TARGETS:
            rows = df_all.query("PandemicPhase == @phase").index
            rows = rows[df_all.loc[rows, tgt].notna()]
            if len(rows) == 0:
                continue
            k = min(N_SAMPLE, len(rows))
            sample_idx[(phase, tgt)] = np.sort(RNG.choice(rows, k, replace=False))

    for phase in PHASES:
        for tgt in TARGETS:
            rows = sample_idx.get((phase, tgt))
            if rows is None:
                continue
            X_raw = df_all.loc[rows].drop(columns=DROP_COLS, errors="ignore")

            mdl_file = MODEL_DIR / f"catboost_{phase}_{tgt}.joblib"
            if not mdl_file.exists():
                continue

            stem = f"CATBOOST_{phase}_{tgt}"
            png_path = PNG_DIR / f"shap_{stem}.png"
            npy_path = NPY_DIR / f"shapvals_{stem}.npy"
            csv_path = CSV_DIR / f"shap_{stem}.csv"

            if png_path.exists() and npy_path.exists() and csv_path.exists():
                continue

            pipe = joblib.load(mdl_file)
            imp = pipe.named_steps["imp"]
            X_fin = imp.transform(X_raw.reindex(columns=imp.feature_names_in_, fill_value=np.nan))
            explainer = shap.TreeExplainer(pipe.named_steps["model"])
            sv = explainer.shap_values(X_fin, check_additivity=False)
            if isinstance(sv, list):
                sv = sv[0]

            plt.figure(figsize=(2.6, 3.2))
            shap.summary_plot(sv, X_fin, feature_names=list(imp.feature_names_in_),
                              max_display=TOP_K, plot_type="dot", show=False)
            plt.title(f"{tgt} | {phase} | CATBOOST", pad=6)
            plt.tight_layout()
            plt.savefig(png_path, bbox_inches="tight")
            plt.close()

            np.save(npy_path, sv)
            pd.DataFrame(sv, columns=imp.feature_names_in_).to_csv(csv_path, index=False, float_format="%.6g")

# -----------------------------------------------------------------------------
# Run All
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_tree_models()
    train_tabpfn()
    generate_shap()
