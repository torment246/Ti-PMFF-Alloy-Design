"""Input schema normalization shared by Ti-PMFF components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELEMENT_COLS, PROCESS_COL, TRANSITION_TEMP_COL


def calculate_t_beta(row: pd.Series) -> float:
    # Supplementary Table S2; concentrations are in wt.% and temperature in °C.
    return float(
        882.0
        + 21.1 * row.get("Al (wt%)", 0)
        + 23.3 * row.get("O (wt%)", 0)
        + 27.8 * row.get("N (wt%)", 0)
        + 17.5 * row.get("C (wt%)", 0)
        - 9.5 * row.get("Mo (wt%)", 0)
        - 4.2 * row.get("V (wt%)", 0)
        - 11.9 * row.get("Fe (wt%)", 0)
        - 14.0 * row.get("Cr (wt%)", 0)
        - 17.5 * row.get("Ni (wt%)", 0)
        - 3.3 * row.get("Sn (wt%)", 0)
        - 1.2 * row.get("Zr (wt%)", 0)
    )


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ELEMENT_COLS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    other_cols = [col for col in ELEMENT_COLS if col != "Ti (wt%)"]
    other_sum = out[other_cols].sum(axis=1)
    ti_missing = out["Ti (wt%)"].isna() | (out["Ti (wt%)"] <= 0)
    out.loc[ti_missing, "Ti (wt%)"] = 100.0 - other_sum.loc[ti_missing]

    if TRANSITION_TEMP_COL not in out.columns:
        out[TRANSITION_TEMP_COL] = np.nan
    out[TRANSITION_TEMP_COL] = pd.to_numeric(out[TRANSITION_TEMP_COL], errors="coerce")
    needs_t_beta = out[TRANSITION_TEMP_COL].isna() | (out[TRANSITION_TEMP_COL] == 0)
    if needs_t_beta.any():
        out.loc[needs_t_beta, TRANSITION_TEMP_COL] = out.loc[needs_t_beta].apply(calculate_t_beta, axis=1)

    if PROCESS_COL not in out.columns:
        out[PROCESS_COL] = ""
    out[PROCESS_COL] = out[PROCESS_COL].fillna("").astype(str)
    return out
