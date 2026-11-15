"""Rank supplied compositions using the GA-selected property models."""
import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from stage1.rerank import rerank
from ti_pmff.config import TARGET_TO_COLUMN
from ti_pmff.prediction import bootstrap_predict

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs/02_prediction/composition"
SELECTION_DIR = ROOT / "outputs/03_feature_selection"
OUTPUT_DIR = ROOT / "outputs/04_composition"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    parser.add_argument("--physical-ranges", type=Path, default=ROOT / "configs/physical_ranges_demo.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-uts", type=float, default=1100.0)
    parser.add_argument("--target-el", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.feature_dir / "train.csv")
    candidates = pd.read_csv(args.feature_dir / "candidates.csv")
    ranges = json.loads(args.physical_ranges.read_text())
    features = {target: pd.read_csv(args.feature_dir / f"{target.lower()}_candidate_fusion.csv")
                for target in TARGET_TO_COLUMN}
    with threadpool_limits(limits=1):
        predictions = bootstrap_predict(train, features, args.selection_dir, "composition",
                                        args.ensemble_size, args.seed)
        ranked = rerank(train, candidates, predictions, ranges, args.target_uts, args.target_el)
    top = ranked.head(args.top_k)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    ranked.to_csv(args.output_dir / "composition_ranked.csv", index=False)
    top.to_csv(args.output_dir / "composition_topk.csv", index=False)
    (args.output_dir / "physical_ranges.json").write_text(json.dumps(ranges, indent=2) + "\n")
    print(top[["rank", "candidate_id", "mean_UTS", "mean_EL", "final_score"]].to_string(index=False))
    print(f"Composition results: {args.output_dir}")


if __name__ == "__main__":
    main()
