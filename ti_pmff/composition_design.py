"""Composition-only inputs and supplied screening candidates."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ELEMENT_COLS
from .schema import prepare_base_dataframe

NEUTRAL_PROCESS = "Processing condition not specified for composition-stage screening"


def prepare_composition_only(data):
    data = prepare_base_dataframe(data)
    data["Process"] = NEUTRAL_PROCESS
    return data


def load_candidate_pool(path: Path):
    data = pd.read_csv(path)
    if not data["candidate_id"].is_unique:
        raise ValueError("Candidate IDs must be unique")
    columns = [column for column in ELEMENT_COLS if column in data]
    weights = data[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(weights).all().all() or (weights < 0).any().any():
        raise ValueError("Candidate fractions must be finite and nonnegative")
    if not np.allclose(weights.sum(axis=1), 100.0, atol=0.01):
        raise ValueError("Candidate fractions must sum to 100 wt.%")
    return prepare_composition_only(data)
