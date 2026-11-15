"""Select fused features using GA, with optional forward/backward comparisons."""
import argparse
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from ga.feature_selection import run_task

ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "outputs/02_prediction"
CONFIG_FILE = ROOT / "configs/ga_demo.json"
OUTPUT_DIR = ROOT / "outputs/03_feature_selection"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["workers"] = args.workers
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Feature rows must follow the row order of the corresponding property CSV.
    tables = []
    with threadpool_limits(limits=1):
        for task in config["tasks"]:
            stage = "composition" if task["stage"] == "composition" else "processing"
            directory = args.feature_dir / stage
            name = task["target"].lower()
            task.update({"data": str(directory / "train.csv"), "test_data": str(directory / "test.csv"),
                         "features": str(directory / f"{name}_train_fusion.csv"),
                         "test_features": str(directory / f"{name}_test_fusion.csv")})
            output = args.output_dir / task["name"]
            print(f"Task: {task['name']}\nData: {task['data']}\nFeatures: {task['features']}")
            run_task(task, config, output)
            tables.append(pd.read_csv(output / "metrics.csv"))
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    print(metrics[["task", "method", "feature_count", "train_r2", "test_r2"]].to_string(index=False))
    print(f"Feature-selection results: {args.output_dir}")


if __name__ == "__main__":
    main()
