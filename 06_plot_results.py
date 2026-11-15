"""Export two-stage analysis figures S6/S7 and tables S5/S6."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "outputs"
OUTPUT_DIR = ROOT / "outputs/06_results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Root containing the Step 04-05 output directories")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ti_pmff.figures import export_figures_and_tables
    export_figures_and_tables(args.results_dir, args.output_dir)
    print(f"Figures (PNG): {args.output_dir / 'figures'}")
    print(f"Tables (CSV): {args.output_dir / 'tables'}")


if __name__ == "__main__":
    main()
