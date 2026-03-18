"""
train.py
--------
Standalone training script.
Run this to pre-train all models before starting the API server.

Usage
-----
    # Train all 43 states
    python train.py

    # Train specific states only
    python train.py --states "California" "Texas" "New York"

    # Override forecast horizon
    python train.py --n-forecast 12
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sales forecasting models for US states."
    )
    parser.add_argument(
        "--data-path",
        default="data/Forecasting_Case-_Study.xlsx",
        help="Path to the Excel dataset",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to save trained model artefacts",
    )
    parser.add_argument(
        "--n-forecast",
        type=int,
        default=8,
        help="Number of weeks to forecast (default: 8)",
    )
    parser.add_argument(
        "--states",
        nargs="*",
        default=None,
        help="Specific states to train (default: all states)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from src.forecaster import SalesForecastingSystem

    system = SalesForecastingSystem(
        data_path     = args.data_path,
        artifacts_dir = args.artifacts_dir,
        n_forecast    = args.n_forecast,
    )

    logger.info("Loading dataset …")
    system.load_data()

    target_states = args.states if args.states else system.states
    logger.info(f"Training {len(target_states)} state(s): {target_states}")

    t_start = time.time()
    success, failed = [], []

    for state in target_states:
        try:
            t0 = time.time()
            result = system.train_state(state)
            elapsed = time.time() - t0
            logger.info(
                f"✅ [{state}] Best: {result.best_model} | "
                f"RMSE={result.all_metrics[result.best_model]['rmse']:.0f} | "
                f"{elapsed:.1f}s"
            )
            success.append(state)
        except Exception as e:
            logger.error(f"❌ [{state}] Failed: {e}")
            failed.append(state)

    total_time = time.time() - t_start

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'STATE':<25} {'BEST MODEL':<12} {'RMSE':>15} {'MAPE':>8}")
    print("-" * 70)
    df = system.summary()
    for _, row in df.iterrows():
        print(f"{row['state']:<25} {row['best_model']:<12} {row['rmse']:>15,.0f} {row['mape']:>7.2f}%")
    print("=" * 70)
    print(f"Trained: {len(success)} ✅  |  Failed: {len(failed)} ❌  |  Time: {total_time:.1f}s")
    if failed:
        print(f"Failed states: {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()
