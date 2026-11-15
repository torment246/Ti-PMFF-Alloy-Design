"""Run the Ti-PMFF demo and optional supplementary figure exports."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Ti-BERT device")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--figures", action="store_true", help="Export Figs. S6/S7 and Tables S5/S6")
    parser.add_argument("--workers", type=int, default=1, help="Workers for XGBoost feature selection")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    bert_dir = output / "01_ti_bert"
    feature_dir = output / "02_prediction"
    selection_dir = output / "03_feature_selection"

    commands = [
        ["01_train_ti_bert.py", "--device", args.device, "--gpu", args.gpu,
         "--output-dir", bert_dir],
        ["02_predict_properties.py", "--bert-device", args.device,
         "--checkpoint", bert_dir / "model.ckpt-8", "--output-dir", feature_dir],
        ["03_select_features.py", "--feature-dir", feature_dir, "--workers", args.workers,
         "--output-dir", selection_dir],
        ["04_design_composition.py", "--feature-dir", feature_dir / "composition",
         "--selection-dir", selection_dir,
         "--output-dir", output / "04_composition"],
        ["05_optimize_processing.py", "--feature-dir", feature_dir / "processing",
         "--selection-dir", selection_dir, "--bert-device", args.device,
         "--composition-results", output / "04_composition/composition_topk.csv",
         "--output-dir", output / "05_processing"],
    ]
    if args.figures:
        commands.append(["06_plot_results.py", "--results-dir", output,
                         "--output-dir", output / "06_results"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu if args.device == "cuda" else "-1"
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ti-pmff-matplotlib"))
    for command in commands:
        print(f"Running {command[0]}", flush=True)
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
