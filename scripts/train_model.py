"""Train and evaluate all models from the command line (no GUI needed).

Usage (from the project folder):
    python scripts/train_model.py

Prints the metrics used in the report and saves the model to models/,
so the Streamlit app starts instantly afterwards.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data_loader import load_dataset  # noqa: E402
from src.model import MAIN_MODEL_NAME, save_bundle, train_models  # noqa: E402


def main() -> None:
    df, info = load_dataset()
    print(f"Dataset: {info.path.name}")
    print(f"Rows in file: {info.raw_rows}   rows after cleaning: {info.clean_rows}   "
          f"duplicates removed: {info.duplicates_removed}")

    bundle = train_models(df, info.fingerprint,
                          progress=lambda f, msg: print(f"  [{f * 100:3.0f}%] {msg}"))
    save_bundle(bundle)

    print(f"\nTraining samples: {bundle['n_train']}   testing samples: {bundle['n_test']}")
    print("\nModel comparison (test set; CV = 5-fold on all rows):")
    with pd.option_context("display.width", 140, "display.float_format", "{:,.3f}".format):
        print(bundle["comparison"].to_string(index=False))
    m = bundle["metrics"]
    print(f"\n{MAIN_MODEL_NAME}: MAE {m['MAE']:.3f}   RMSE {m['RMSE']:.3f}   R2 {m['R2']:.3f}")
    print(f"Best model by test RMSE: {bundle['best_by_rmse']}")
    print(f"Prediction-range coverage on test set: {bundle['interval_coverage'] * 100:.1f}%")
    print("\nFeature importance (grouped):")
    print(bundle["feature_importance"].to_string(index=False))
    print(f"\nSaved model to models/ in {bundle['meta']['training_seconds']} s.")


if __name__ == "__main__":
    main()
