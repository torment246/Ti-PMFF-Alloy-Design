"""Bootstrap predictions using the GA-selected features and regressor settings."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import TARGET_TO_COLUMN


def bootstrap_predict(train, candidate_features, selection_dir, stage, ensemble_size=5, seed=42):
    rng = np.random.default_rng(seed)
    result = pd.DataFrame()
    for target, column in TARGET_TO_COLUMN.items():
        selected = joblib.load(Path(selection_dir) / f"{stage}_{target}/GA_model.joblib")
        mask = selected["mask"]
        task = selected["task"]
        features = pd.read_csv(task["features"])
        y = pd.to_numeric(train[column], errors="coerce")
        valid = y.notna() & (y > 0)
        X = features.loc[valid].to_numpy(dtype=float)[:, mask]
        y = y.loc[valid].to_numpy(dtype=float)
        candidates = candidate_features[target].to_numpy(dtype=float)[:, mask]
        predictions = []
        for _ in range(ensemble_size):
            rows = rng.integers(0, len(y), size=len(y))
            model = clone(selected["model"])
            model.fit(X[rows], y[rows])
            predictions.append(model.predict(candidates))
        predictions = np.asarray(predictions)
        result[f"mean_{target}"] = predictions.mean(axis=0)
        result[f"std_{target}"] = predictions.std(axis=0)
        result[f"selected_features_{target}"] = int(mask.sum())
    if stage == "process":
        result["raw_mean_EL"] = result["mean_EL"]
        result["mean_EL"] = result["mean_EL"].clip(lower=0.0)
    return result
