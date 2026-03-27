from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ..utils.metrics import classification_metrics

def _save_model(model, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{name}.joblib")

def train_eval_models(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    out_model_dir: Path,
    run_info: dict,
):
    rows = []
    preds_for_roc = []

    models = {
        "RF": RandomForestClassifier(
            n_estimators=800, max_depth=None, n_jobs=-1, random_state=run_info["seed"],
            class_weight="balanced_subsample"
        ),
        "XGB": XGBClassifier(
            n_estimators=1200, max_depth=6, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, objective="binary:logistic",
            eval_metric="auc", n_jobs=-1, random_state=run_info["seed"]
        ),
        "LGBM": LGBMClassifier(
            n_estimators=2000, learning_rate=0.03,
            num_leaves=63, subsample=0.9, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=run_info["seed"],
            class_weight="balanced"
        )
    }

    for name, model in models.items():
        # For tree models on descriptors, scaling is not required; but safe to include for consistency if dense input.
        # If X is sparse, StandardScaler(with_mean=False) works.
        scaler = StandardScaler(with_mean=False)
        pipe = Pipeline([("scaler", scaler), ("model", model)])
        pipe.fit(X_train, y_train)

        # choose threshold=0.5 for now; can tune later
        prob_test = pipe.predict_proba(X_test)[:, 1]
        m = classification_metrics(y_test, prob_test, threshold=0.5)

        row = dict(run_info)
        row.update({
            "model": name,
            "split": "test",
            **asdict(m)
        })
        rows.append(row)

        _save_model(pipe, out_model_dir, f"{name}_seed{run_info['seed']}")
        preds_for_roc.append({"name": name, "y_true": y_test, "y_prob": prob_test})

    return rows, preds_for_roc
