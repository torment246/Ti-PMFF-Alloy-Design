"""Composition ranking using performance, physical feasibility and distance."""
import numpy as np
import pandas as pd

from ti_pmff.config import ELEMENT_COLS
from ti_pmff.physical_features import compute_physical_features

SCORE_FORMULA = "0.55*performance + 0.30*physical_feasibility + 0.10*confidence + 0.05*diversity"


def minmax(values):
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 1e-12 else np.zeros_like(values)


def rerank(train, candidates, predictions, physical_ranges, target_uts=1100.0, target_el=16.0):
    ranked = pd.concat([candidates.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    train_phy = compute_physical_features(train)
    candidate_phy = compute_physical_features(candidates)
    for column in candidate_phy:
        ranked[column] = candidate_phy[column].to_numpy()

    train_values = pd.concat([train[ELEMENT_COLS].reset_index(drop=True), train_phy.reset_index(drop=True)], axis=1)
    candidate_values = pd.concat([candidates[ELEMENT_COLS].reset_index(drop=True), candidate_phy.reset_index(drop=True)], axis=1)
    center = train_values.median()
    scale = train_values.std().replace(0, 1.0).fillna(1.0)
    train_scaled = ((train_values - center) / scale).to_numpy()
    candidate_scaled = ((candidate_values - center) / scale).to_numpy()
    distance = np.linalg.norm(candidate_scaled[:, None, :] - train_scaled[None, :, :], axis=2).min(axis=1)
    ranked["nearest_train_distance"] = distance
    # Normalize over the candidate pool; confidence is the complement of diversity.
    ranked["confidence_component"] = 1.0 - minmax(distance)
    ranked["diversity_component"] = minmax(distance)

    feasible = []
    for name in ["MoEq", "AlEq", "T_beta"]:
        lower, upper = physical_ranges[name]
        inside = ranked[name].between(lower, upper, inclusive="both")
        ranked[f"{name}_feasible"] = inside
        feasible.append(inside.to_numpy(dtype=float))
    ranked["physical_feasibility_component"] = np.mean(feasible, axis=0)

    ranked["meets_UTS_target"] = ranked["mean_UTS"] >= target_uts
    ranked["meets_EL_target"] = ranked["mean_EL"] >= target_el
    ranked["meets_both_targets"] = ranked["meets_UTS_target"] & ranked["meets_EL_target"]
    performance = 0.5 * minmax(ranked["mean_UTS"]) + 0.5 * minmax(ranked["mean_EL"])
    performance += (0.15 * ranked["meets_UTS_target"] + 0.15 * ranked["meets_EL_target"]
                    + 0.20 * ranked["meets_both_targets"])
    ranked["performance_component"] = np.clip(performance, 0, 1)
    ranked["final_score"] = (
        0.55 * ranked["performance_component"]
        + 0.30 * ranked["physical_feasibility_component"]
        + 0.10 * ranked["confidence_component"]
        + 0.05 * ranked["diversity_component"]
    )
    ranked["score_formula"] = SCORE_FORMULA
    ranked = ranked.sort_values("final_score", ascending=False, kind="stable").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked
