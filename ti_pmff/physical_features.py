"""Composition-derived physical descriptors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph_features import ELEMENT_PROPS
from .schema import calculate_t_beta, prepare_base_dataframe

ATOMIC_WEIGHTS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "Al": 26.9815385, "Si": 28.085, "Ti": 47.867, "V": 50.9415,
    "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Y": 88.90584, "Zr": 91.224,
    "Nb": 92.90637, "Mo": 95.95, "Sn": 118.710, "Hf": 178.49,
    "Ta": 180.94788, "W": 183.84,
}
PHYSICAL_NAMES = ["MoEq", "AlEq", "VEC", "delta_R", "T_beta"]


def atomic_fractions(df: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(index=df.index)
    for element in ATOMIC_WEIGHTS:
        column = f"{element} (wt%)"
        weights[element] = pd.to_numeric(df[column], errors="coerce").fillna(0.0) if column in df else 0.0

    for column in df.filter(regex=r"\(wt%\)$"):
        element = column.removesuffix(" (wt%)")
        if element not in ATOMIC_WEIGHTS and element != "TiB":
            if pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0).any():
                raise ValueError(f"Missing atomic properties for {element}")

    # TiB contributes one Ti atom and one B atom per formula unit.
    if "TiB (wt%)" in df:
        tib = pd.to_numeric(df["TiB (wt%)"], errors="coerce").fillna(0.0)
        if tib.lt(0).any():
            raise ValueError("TiB content must be nonnegative")
        formula_mass = ATOMIC_WEIGHTS["Ti"] + ATOMIC_WEIGHTS["B"]
        weights["Ti"] += tib * ATOMIC_WEIGHTS["Ti"] / formula_mass
        weights["B"] += tib * ATOMIC_WEIGHTS["B"] / formula_mass

    if not np.isfinite(weights.to_numpy()).all() or weights.lt(0).any().any():
        raise ValueError("Element contents must be finite and nonnegative")
    moles = weights.div(pd.Series(ATOMIC_WEIGHTS))
    total = moles.sum(axis=1)
    if total.le(0).any():
        raise ValueError("A composition must contain positive elemental mass")
    return moles.div(total, axis=0)


def compute_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_base_dataframe(df)
    out = pd.DataFrame(index=prepared.index)
    out["MoEq"] = (
        prepared.get("Mo (wt%)", 0)
        + 0.67 * prepared.get("V (wt%)", 0)
        + 0.44 * prepared.get("W (wt%)", 0)
        + 0.28 * prepared.get("Nb (wt%)", 0)
        + 0.22 * prepared.get("Ta (wt%)", 0)
        + 2.9 * prepared.get("Fe (wt%)", 0)
        + 1.6 * prepared.get("Cr (wt%)", 0)
    )
    out["AlEq"] = (
        prepared.get("Al (wt%)", 0)
        + prepared.get("Sn (wt%)", 0) / 3
        + prepared.get("Zr (wt%)", 0) / 6
        + 10 * (prepared.get("O (wt%)", 0) + prepared.get("C (wt%)", 0) + 2 * prepared.get("N (wt%)", 0))
    )
    # VEC and size mismatch use atomic fractions, not wt.%. Delta_R is dimensionless.
    fractions = atomic_fractions(prepared)
    valence = pd.Series({element: ELEMENT_PROPS[element][2] for element in ATOMIC_WEIGHTS})
    radii = pd.Series({element: ELEMENT_PROPS[element][0] for element in ATOMIC_WEIGHTS})
    out["VEC"] = fractions.mul(valence).sum(axis=1)
    mean_radius = fractions.mul(radii).sum(axis=1)
    relative_radius = pd.DataFrame(
        np.tile(radii.to_numpy(), (len(prepared), 1)), index=prepared.index, columns=radii.index
    ).div(mean_radius, axis=0)
    out["delta_R"] = np.sqrt(fractions.mul((1.0 - relative_radius).pow(2)).sum(axis=1))
    out["T_beta"] = prepared.apply(calculate_t_beta, axis=1)
    return out[PHYSICAL_NAMES]

