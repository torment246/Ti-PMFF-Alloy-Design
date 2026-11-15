"""Scoring functions for joint composition-process candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .process_parser import parse_process
from ti_pmff.physical_features import PHYSICAL_NAMES, compute_physical_features
from ti_pmff.schema import prepare_base_dataframe

COMPOSITION_FEATURES = ["Ti (wt%)", "Al (wt%)", "V (wt%)", "Mo (wt%)", "Fe (wt%)", "Sn (wt%)", *PHYSICAL_NAMES]
PROCESS_FEATURES = [
    "initial_temp",
    "initial_deformation",
    "secondary_temp",
    "passes",
    "total_cyclic_deformation",
    "final_total_deformation",
    "secondary_time",
    "deformation_per_pass",
]
MODE_WEIGHTS = {
    "balanced": (0.30, 0.40, 0.20, 0.10),
    "exploration": (0.45, 0.20, 0.30, 0.05),
    "exploitation": (0.15, 0.65, 0.10, 0.10),
}


def minmax_norm(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    if hi - lo < 1e-12:
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return out


def _with_physical(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_base_dataframe(df)
    physical = compute_physical_features(out)
    out[physical.columns] = physical
    return out


def _train_process_features(train_df: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame([parse_process(value) for value in train_df.get("Process", "")])
    out = pd.DataFrame(
        {
            "initial_temp": parsed["initial_forging_temp"],
            "initial_deformation": parsed["initial_deformation_pct"],
            "secondary_temp": parsed["secondary_forging_temp"],
            "passes": parsed["passes"],
            "total_cyclic_deformation": parsed["total_cyclic_deformation_pct"],
            "final_total_deformation": parsed["final_total_deformation_pct"],
            "secondary_time": parsed["secondary_holding_time_min"],
            "deformation_per_pass": parsed["deformation_per_pass_pct"],
        }
    )
    for col in out.columns[out.isna().all(axis=0)]:
        out[col] = 0.0
    return out


def _distance_to_train(candidate: pd.DataFrame, train: pd.DataFrame, cols: list[str]) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(imputer.fit_transform(train[cols]))
    cand_arr = scaler.transform(imputer.transform(candidate[cols]))
    return pairwise_distances(cand_arr, train_arr).min(axis=1)


def _composition_penalty(
    row: pd.Series,
    comp_stats: pd.DataFrame,
    physical_q05: pd.Series,
    physical_q95: pd.Series,
    sn_range: tuple[float, float] = (2.0, 3.0),
    min_effective_mo: float = 0.3,
    min_effective_fe: float = 0.3,
) -> tuple[float, str]:
    penalty = 0.0
    violations = []
    elements = ["Ti (wt%)", "Al (wt%)", "V (wt%)", "Mo (wt%)", "Fe (wt%)", "Sn (wt%)"]
    if row["Ti (wt%)"] <= 0:
        penalty += 0.5
        violations.append("Ti非正")
    if row["Al (wt%)"] <= 0:
        penalty += 0.3
        violations.append("Al非正")
    if row["V (wt%)"] <= 0:
        penalty += 0.3
        violations.append("V非正")
    if row["Mo (wt%)"] <= 0 or row["Mo (wt%)"] < min_effective_mo:
        penalty += 0.3
        violations.append("Mo低于有效下限")
    if row["Fe (wt%)"] <= 0 or row["Fe (wt%)"] < min_effective_fe:
        penalty += 0.3
        violations.append("Fe低于有效下限")
    if row["Sn (wt%)"] <= 0:
        penalty += 0.3
        violations.append("Sn非正")
    if any(float(row[col]) < 0 for col in elements):
        penalty += 0.5
        violations.append("元素含量为负")
    if abs(sum(float(row[col]) for col in elements) - 100.0) > 0.5:
        penalty += 0.3
        violations.append("元素总和偏离100")
    stats = comp_stats.set_index("element")
    for col in ["Al (wt%)", "V (wt%)", "Mo (wt%)", "Fe (wt%)", "Sn (wt%)"]:
        if col == "Sn (wt%)" and sn_range[0] <= float(row[col]) <= sn_range[1]:
            continue
        if col in stats.index:
            val = float(row[col])
            span = max(float(stats.loc[col, "q95"] - stats.loc[col, "q05"]), 1.0)
            if val < float(stats.loc[col, "q05"]) - span or val > float(stats.loc[col, "q95"]) + span:
                penalty += 0.05
                violations.append(f"{col}超出训练分位范围")
    beta_stabilizer_sum = float(row["V (wt%)"]) + float(row["Mo (wt%)"]) + float(row["Fe (wt%)"])
    mo_fe_sum = float(row["Mo (wt%)"]) + float(row["Fe (wt%)"])
    if mo_fe_sum < 1.0:
        penalty += 0.05
        violations.append("Mo+Fe低于1.0")
    if beta_stabilizer_sum > 0 and float(row["V (wt%)"]) / beta_stabilizer_sum > 0.85:
        penalty += 0.05
        violations.append("V在beta稳定元素中过度主导")
    for col in PHYSICAL_NAMES:
        val = float(row[col])
        span = max(float(physical_q95[col] - physical_q05[col]), 1.0)
        if val < float(physical_q05[col]) - span or val > float(physical_q95[col]) + span:
            penalty += 0.05
            violations.append(f"{col}超出训练合理范围")
    return min(penalty, 1.0), "; ".join(violations) if violations else "无明显成分约束风险"


def _process_penalty(row: pd.Series) -> tuple[float, str]:
    penalty = 0.0
    violations = []
    checks = [
        ("initial_temp", "初始温度非正", lambda v: v <= 0, 0.3),
        ("secondary_temp", "二次温度非正", lambda v: v <= 0, 0.3),
        ("initial_time", "初始时间非正", lambda v: v <= 0, 0.2),
        ("secondary_time", "二次时间非正", lambda v: v <= 0, 0.2),
        ("deformation_per_pass", "单道变形非正", lambda v: v <= 0, 0.2),
        ("passes", "passes非正", lambda v: v <= 0, 0.2),
        ("total_cyclic_deformation", "循环变形不在0-100", lambda v: v < 0 or v > 100, 0.3),
        ("final_total_deformation", "最终变形不在0-100", lambda v: v < 0 or v > 100, 0.3),
    ]
    for col, label, fn, value_penalty in checks:
        val = float(row.get(col, np.nan))
        if not np.isfinite(val) or fn(val):
            penalty += value_penalty
            violations.append(label)
    if float(row["final_total_deformation"]) < float(row["initial_deformation"]):
        penalty += 0.1
        violations.append("最终变形小于初始变形")
    if float(row["final_total_deformation"]) < float(row["total_cyclic_deformation"]):
        penalty += 0.05
        violations.append("最终变形小于循环总变形")
    return min(penalty, 1.0), "; ".join(violations) if violations else "无明显工艺约束风险"


def score_joint_candidates(
    candidate_df: pd.DataFrame,
    train_df: pd.DataFrame,
    predictions,
    comp_stats: pd.DataFrame,
    mode: str = "balanced",
    target_uts: float = 1500.0,
    target_el: float = 10.0,
    sn_range: tuple[float, float] = (2.0, 3.0),
    min_effective_mo: float = 0.3,
    min_effective_fe: float = 0.3,
) -> pd.DataFrame:
    if mode not in MODE_WEIGHTS:
        raise ValueError(f"Unknown mode: {mode}")
    ranked = _with_physical(candidate_df)
    train_prepared = _with_physical(train_df)
    ranked["mean_UTS"] = predictions.mean_UTS
    ranked["std_UTS"] = predictions.std_UTS
    ranked["mean_EL"] = predictions.mean_EL
    ranked["std_EL"] = predictions.std_EL
    if hasattr(predictions, "raw_mean_EL"):
        ranked["raw_mean_EL"] = predictions.raw_mean_EL

    ranked["uncertainty_score"] = 0.6 * minmax_norm(ranked["std_UTS"]) + 0.4 * minmax_norm(ranked["std_EL"])
    p_base = 0.6 * minmax_norm(ranked["mean_UTS"]) + 0.4 * minmax_norm(ranked["mean_EL"])
    p_bonus = (
        0.1 * (ranked["mean_UTS"] >= target_uts).astype(float)
        + 0.1 * (ranked["mean_EL"] >= target_el).astype(float)
        + 0.2 * ((ranked["mean_UTS"] >= target_uts) & (ranked["mean_EL"] >= target_el)).astype(float)
    )
    ranked["performance_score"] = np.clip(p_base + p_bonus, 0.0, 1.0)

    ranked["nearest_train_composition_distance"] = _distance_to_train(ranked, train_prepared, COMPOSITION_FEATURES)
    ranked["composition_diversity_score"] = minmax_norm(ranked["nearest_train_composition_distance"])

    train_proc = _train_process_features(train_prepared)
    ranked["nearest_train_process_distance"] = _distance_to_train(ranked, train_proc, PROCESS_FEATURES)
    ranked["process_diversity_score"] = minmax_norm(ranked["nearest_train_process_distance"])
    ranked["diversity_score"] = 0.6 * ranked["composition_diversity_score"] + 0.4 * ranked["process_diversity_score"]

    physical_q05 = train_prepared[PHYSICAL_NAMES].quantile(0.05)
    physical_q95 = train_prepared[PHYSICAL_NAMES].quantile(0.95)
    comp_penalties = [
        _composition_penalty(row, comp_stats, physical_q05, physical_q95, sn_range, min_effective_mo, min_effective_fe)
        for _, row in ranked.iterrows()
    ]
    proc_penalties = [_process_penalty(row) for _, row in ranked.iterrows()]
    ranked["composition_constraint_penalty"] = [item[0] for item in comp_penalties]
    ranked["process_constraint_penalty"] = [item[0] for item in proc_penalties]
    ranked["constraint_penalty"] = np.minimum(
        ranked["composition_constraint_penalty"] + ranked["process_constraint_penalty"], 1.0
    )
    ranked["constraint_violations"] = [
        "; ".join([c_msg, p_msg]) for (_, c_msg), (_, p_msg) in zip(comp_penalties, proc_penalties)
    ]

    alpha, beta, gamma, delta = MODE_WEIGHTS[mode]
    ranked["final_score"] = (
        alpha * ranked["uncertainty_score"]
        + beta * ranked["performance_score"]
        + gamma * ranked["diversity_score"]
        - delta * ranked["constraint_penalty"]
    )
    ranked["meets_UTS_target"] = ranked["mean_UTS"] >= target_uts
    ranked["meets_EL_target"] = ranked["mean_EL"] >= target_el
    ranked["meets_both_targets"] = ranked["meets_UTS_target"] & ranked["meets_EL_target"]
    ranked["recommendation_type"] = "未入选"
    ranked["selection_role"] = "候选样本"
    return ranked.sort_values("final_score", ascending=False).reset_index(drop=True)
