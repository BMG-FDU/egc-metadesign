import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from seven_descriptor_shap import FEATURE_NAMES, analyze_seven_descriptors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-table", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--target-column", default="GNN_SEA_J_per_kg")
    parser.add_argument("--fidelity-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input_table)
    missing = [name for name in FEATURE_NAMES + [args.target_column] if name not in data.columns]
    if missing:
        parser.error("Missing columns: " + ", ".join(missing))
    id_column = next((name for name in ("graph_id", "sample_id") if name in data.columns), None)
    analyze_seven_descriptors(
        data[FEATURE_NAMES].to_numpy(float), data[args.target_column].to_numpy(float),
        args.outdir, sample_ids=data[id_column].astype(str).tolist() if id_column else None,
        seed=args.seed, fidelity_fraction=args.fidelity_fraction,
        predictor_path=args.predictor, make_plots=not args.no_plots,
        saved_splits=data['proxy_split'].astype(str).to_numpy() if 'proxy_split' in data.columns else None,
    )


if __name__ == "__main__":
    main()
