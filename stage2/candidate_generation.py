"""Utilities for fixed-composition processing candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ti_pmff.schema import prepare_base_dataframe

RANGE_ELEMENTS = ["Al (wt%)", "V (wt%)", "Mo (wt%)", "Fe (wt%)", "Sn (wt%)"]

PROCESS_TEMPLATE = (
    "Initial Forging: {initial_temp:.1f}°C for {initial_time:.1f}h, "
    "{initial_deformation:g}% deformation. Secondary Forging: {secondary_temp:.1f}°C for "
    "{secondary_time:.1f}min, {deformation_per_pass:g}% deformation per pass, passes: {passes}, "
    "total cyclic deformation: {total_cyclic_deformation:g}%. Final total deformation: "
    "{final_total_deformation:g}%, cooled by {cooling_type}."
)


def normalize_element_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for symbol in ["Ti", "Al", "V", "Mo", "Fe", "Sn"]:
        standard = f"{symbol} (wt%)"
        if standard not in out.columns and symbol in out.columns:
            out[standard] = out[symbol]
    return prepare_base_dataframe(out)


def filter_ti_al_v(df: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_element_columns(df)
    mask = (
        (pd.to_numeric(prepared["Ti (wt%)"], errors="coerce").fillna(0.0) > 0)
        & (pd.to_numeric(prepared["Al (wt%)"], errors="coerce").fillna(0.0) > 0)
        & (pd.to_numeric(prepared["V (wt%)"], errors="coerce").fillna(0.0) > 0)
    )
    return prepared.loc[mask].reset_index(drop=True)


def composition_range_stats(filtered_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in RANGE_ELEMENTS:
        values = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0.0)
        nonzero = values > 0
        nonzero_values = values[nonzero]
        rows.append(
            {
                "element": col,
                "min": float(values.min()),
                "max": float(values.max()),
                "q05": float(values.quantile(0.05)),
                "q95": float(values.quantile(0.95)),
                "nonzero_q05": float(nonzero_values.quantile(0.05)) if len(nonzero_values) else np.nan,
                "nonzero_q95": float(nonzero_values.quantile(0.95)) if len(nonzero_values) else np.nan,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "nonzero_count": int(nonzero.sum()),
                "nonzero_frequency": float(nonzero.mean()),
            }
        )
    return pd.DataFrame(rows)


def process_text(params: dict) -> str:
    return PROCESS_TEMPLATE.format(**params)


def generate_process_pool(design: dict) -> pd.DataFrame:
    composition_id = design["composition_id"]
    composition = design["composition"]
    fixed_process = design["fixed_process"]
    process_space = design["process_space"]
    rows = []
    counter = 1
    for secondary_temp in process_space["secondary_temp"]:
        for passes in process_space["passes"]:
            for cyclic in process_space["total_cyclic_deformation"]:
                for final_def in process_space["final_total_deformation"]:
                    params = dict(fixed_process)
                    params.update(
                        {
                            "secondary_temp": secondary_temp,
                            "passes": int(passes),
                            "total_cyclic_deformation": cyclic,
                            "final_total_deformation": final_def,
                        }
                    )
                    row = {
                        "candidate_id": f"{composition_id}-PROC-{counter:03d}",
                        "composition_id": composition_id,
                        "composition_source": "fixed_composition",
                        "is_fixed_composition": True,
                        **composition,
                        **params,
                    }
                    row["Process"] = process_text(row)
                    rows.append(row)
                    counter += 1
    return normalize_element_columns(pd.DataFrame(rows))
