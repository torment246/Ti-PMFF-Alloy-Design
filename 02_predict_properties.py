"""Split property data, train fusion models and evaluate held-out predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from ga.feature_selection import Evaluator, fit_and_evaluate
from ti_pmff.composition_design import load_candidate_pool, prepare_composition_only
from ti_pmff.fusion_pipeline import TiPMFFFeaturePipeline
from ti_pmff.schema import prepare_base_dataframe

ROOT = Path(__file__).resolve().parent
COMPOSITION_DATA = ROOT / "data/demo/composition.csv"
PROCESSING_DATA = ROOT / "data/demo/processing.csv"
CANDIDATE_FILE = ROOT / "data/demo/candidates.csv"
OUTPUT_DIR = ROOT / "outputs/02_prediction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--composition-data", type=Path, default=COMPOSITION_DATA)
    parser.add_argument("--processing-data", type=Path, default=PROCESSING_DATA)
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_FILE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/01_ti_bert/model.ckpt-8")
    parser.add_argument("--bert-config", type=Path, default=ROOT / "configs/bert_demo.json")
    parser.add_argument("--vocab", type=Path, default=ROOT / "data/demo/vocab.txt")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Fusion training device")
    parser.add_argument("--fusion-epochs", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    datasets = {
        "composition": prepare_composition_only(pd.read_csv(args.composition_data)),
        "processing": prepare_base_dataframe(pd.read_csv(args.processing_data)),
    }
    candidates = load_candidate_pool(args.candidates)
    with threadpool_limits(limits=1):
        for stage, data in datasets.items():
            output = args.output_dir / stage
            output.mkdir(parents=True, exist_ok=True)
            data.insert(0, "source_row", np.arange(len(data)))
            train, test = train_test_split(data, test_size=args.test_size, random_state=args.seed)
            train, test = train.reset_index(drop=True), test.reset_index(drop=True)
            train.to_csv(output / "train.csv", index=False)
            test.to_csv(output / "test.csv", index=False)
            print(f"{stage}: {len(train)} training records, {len(test)} test records")
            model = TiPMFFFeaturePipeline(
                output, args.checkpoint, args.bert_config, args.vocab,
                device=args.device, bert_device=args.bert_device,
                fusion_epochs=args.fusion_epochs, seed=args.seed,
            )
            model.fit_transform(train, test, candidates if stage == "composition" else None)
        candidates.to_csv(args.output_dir / "composition/candidates.csv", index=False)

        config = json.loads((ROOT / "configs/ga_demo.json").read_text())
        rows, predictions = [], []
        for task in config["tasks"]:
            stage = "composition" if task["stage"] == "composition" else "processing"
            directory = args.output_dir / stage
            name = task["target"].lower()
            task.update({"data": str(directory / "train.csv"), "test_data": str(directory / "test.csv"),
                         "features": str(directory / f"{name}_train_fusion.csv"),
                         "test_features": str(directory / f"{name}_test_fusion.csv")})
            evaluator = Evaluator(task, seed=args.seed)
            _, scores, pred = fit_and_evaluate(evaluator, np.ones(len(evaluator.names), dtype=bool), task)
            rows.append({"task": task["name"], "model": task["model"], **scores})
            pred.insert(0, "task", task["name"])
            predictions.append(pred)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(args.output_dir / "prediction_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "prediction_predictions.csv", index=False)
    print(metrics[["task", "test_r2", "test_mae", "test_rmse"]].to_string(index=False))
    print(f"Features and predictions: {args.output_dir}")


if __name__ == "__main__":
    main()
