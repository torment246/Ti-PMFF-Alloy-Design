"""Optimize processing for the first composition recommended by Step 04."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from stage2.candidate_generation import composition_range_stats, filter_ti_al_v, generate_process_pool
from stage2.scoring import score_joint_candidates
from stage2.select_topk import select_topk
from ti_pmff.config import ELEMENT_COLS
from ti_pmff.fusion_pipeline import TiPMFFFeaturePipeline
from ti_pmff.prediction import bootstrap_predict

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs/02_prediction/processing"
SELECTION_DIR = ROOT / "outputs/03_feature_selection"
COMPOSITION_FILE = ROOT / "outputs/04_composition/composition_topk.csv"
OUTPUT_DIR = ROOT / "outputs/05_processing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-results", type=Path, default=COMPOSITION_FILE)
    parser.add_argument("--design-config", type=Path, default=ROOT / "configs/design_demo.json")
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--selection-dir", type=Path, default=SELECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bert-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-uts", type=float, default=1500.0)
    parser.add_argument("--target-el", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    selected = pd.read_csv(args.composition_results).iloc[0]
    design = json.loads(args.design_config.read_text())
    design["composition_id"] = str(selected["candidate_id"])
    design["composition"] = {column: float(selected.get(column, 0)) for column in ELEMENT_COLS}
    design["targets"] = {"UTS": args.target_uts, "EL": args.target_el}
    candidates = generate_process_pool(design)
    candidates.to_csv(args.output_dir / "process_candidates.csv", index=False)
    (args.output_dir / "design_settings.json").write_text(json.dumps(design, indent=2) + "\n")
    train = pd.read_csv(args.feature_dir / "train.csv")

    with threadpool_limits(limits=1):
        # Transform the new routes with the fusion models fitted in Step 02.
        pipeline = TiPMFFFeaturePipeline.from_directory(args.feature_dir, args.device, args.bert_device)
        features = pipeline.transform_candidates(candidates, args.output_dir / "features")
        predictions = bootstrap_predict(train, features, args.selection_dir, "process",
                                        args.ensemble_size, args.seed)
        statistics = composition_range_stats(filter_ti_al_v(train))
        ranked = score_joint_candidates(
            candidates, train, predictions, statistics, mode="balanced",
            target_uts=args.target_uts, target_el=args.target_el,
            sn_range=(1.5, 3.0), min_effective_mo=0.3, min_effective_fe=0.3,
        )
        top = select_topk(ranked, args.top_k)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    ranked.to_csv(args.output_dir / "process_ranked.csv", index=False)
    top.to_csv(args.output_dir / "process_topk.csv", index=False)
    print(f"Fixed composition: {design['composition_id']}")
    print(top[["rank", "candidate_id", "mean_UTS", "mean_EL", "final_score"]].to_string(index=False))
    print(f"Processing results: {args.output_dir}")


if __name__ == "__main__":
    main()
