"""Select the highest-scoring processing routes."""
import pandas as pd


def select_topk(ranked: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    out = ranked.sort_values("final_score", ascending=False, kind="stable").head(top_k).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out["recommendation_type"] = "Top-K processing candidate"
    columns = [
        "rank", "recommendation_type", "candidate_id", "composition_id",
        "Ti (wt%)", "Al (wt%)", "V (wt%)", "Mo (wt%)", "Fe (wt%)", "Sn (wt%)",
        "mean_UTS", "std_UTS", "mean_EL", "std_EL", "final_score",
        "secondary_temp", "passes", "total_cyclic_deformation", "final_total_deformation",
        "meets_UTS_target", "meets_EL_target", "meets_both_targets", "Process",
    ]
    return out[columns]
