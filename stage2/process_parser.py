"""Parse heat-treatment and forging text into numeric process features."""

from __future__ import annotations

import re
from typing import Dict

import numpy as np


def _first_float(pattern: str, text: str) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return np.nan


def parse_process(process: object) -> Dict[str, float]:
    """Parse forging and cooling features; leave missing fields unspecified."""

    text = "" if process is None else str(process)
    lower = text.lower()

    features = {
        "initial_forging_temp": np.nan,
        "initial_holding_time_h": np.nan,
        "initial_deformation_pct": np.nan,
        "secondary_forging_temp": np.nan,
        "secondary_holding_time_min": np.nan,
        "deformation_per_pass_pct": np.nan,
        "passes": np.nan,
        "total_cyclic_deformation_pct": np.nan,
        "final_total_deformation_pct": np.nan,
        "cooling_AC": 0.0,
        "cooling_WQ": 0.0,
        "cooling_FC": 0.0,
        "cooling_OQ": 0.0,
        "process_parse_quality": 0.0,
    }
    if not text.strip():
        return features

    initial_block = re.search(
        r"Initial\s+Forging\s*:\s*([0-9.]+)\s*°?C\s*for\s*([0-9.]+)\s*h\s*,\s*([0-9.]+)\s*%\s*deformation",
        text,
        flags=re.IGNORECASE,
    )
    if initial_block:
        features["initial_forging_temp"] = float(initial_block.group(1))
        features["initial_holding_time_h"] = float(initial_block.group(2))
        features["initial_deformation_pct"] = float(initial_block.group(3))
    else:
        features["initial_forging_temp"] = _first_float(
            r"(?:Solution|Forging|Annealing)\s*:\s*([0-9.]+)\s*°?C", text
        )
        features["initial_holding_time_h"] = _first_float(
            r"(?:Solution|Forging|Annealing)\s*:\s*[0-9.]+\s*°?C\s*for\s*([0-9.]+)\s*h",
            text,
        )

    secondary_block = re.search(
        r"Secondary\s+Forging\s*:\s*([0-9.]+)\s*°?C\s*for\s*([0-9.]+)\s*min\s*,\s*([0-9.]+)\s*%\s*deformation\s+per\s+pass\s*,\s*passes\s*:\s*([0-9.]+)\s*,\s*total\s+cyclic\s+deformation\s*:\s*([0-9.]+)\s*%",
        text,
        flags=re.IGNORECASE,
    )
    if secondary_block:
        features["secondary_forging_temp"] = float(secondary_block.group(1))
        features["secondary_holding_time_min"] = float(secondary_block.group(2))
        features["deformation_per_pass_pct"] = float(secondary_block.group(3))
        features["passes"] = float(secondary_block.group(4))
        features["total_cyclic_deformation_pct"] = float(secondary_block.group(5))
    else:
        features["secondary_forging_temp"] = _first_float(
            r"(?:Aging|Secondary\s+Forging)\s*:\s*([0-9.]+)\s*°?C", text
        )
        aging_h = _first_float(r"Aging\s*:\s*[0-9.]+\s*°?C\s*for\s*([0-9.]+)\s*h", text)
        if not np.isnan(aging_h):
            features["secondary_holding_time_min"] = aging_h * 60.0

    features["final_total_deformation_pct"] = _first_float(
        r"Final\s+total\s+deformation\s*:\s*([0-9.]+)\s*%", text
    )

    if "air cool" in lower or "(ac)" in lower:
        features["cooling_AC"] = 1.0
    if "water quench" in lower or "(wq)" in lower:
        features["cooling_WQ"] = 1.0
    if "furnace cool" in lower or "(fc)" in lower:
        features["cooling_FC"] = 1.0
    if "oil quench" in lower or "(oq)" in lower:
        features["cooling_OQ"] = 1.0

    expected_numeric = [
        "initial_forging_temp",
        "initial_holding_time_h",
        "initial_deformation_pct",
        "secondary_forging_temp",
        "secondary_holding_time_min",
        "deformation_per_pass_pct",
        "passes",
        "total_cyclic_deformation_pct",
        "final_total_deformation_pct",
    ]
    parsed = sum(not np.isnan(features[name]) for name in expected_numeric)
    parsed += sum(features[name] > 0 for name in ["cooling_AC", "cooling_WQ", "cooling_FC", "cooling_OQ"])
    features["process_parse_quality"] = parsed / (len(expected_numeric) + 4)
    return features
