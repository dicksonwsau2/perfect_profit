"""
perfect_profit.py

Provides the main orchestrator for Perfect Profit analysis, including:
 - The `run_pp_analysis` function for analyzing tradeplans,
 - A `main_cli` function to parse command-line arguments and call `run_pp_analysis`,
 - An `if __name__ == "__main__": main_cli()` entry point.

All concurrency, dataset reading, correlation, etc. happen inside run_pp_analysis or
its helper modules (pp_compute, correlation_manager, etc.).
"""

import argparse
import logging
import sys
from typing import List
from pp_compute import compute_pp_curves_for_dataset
from correlation_manager import compare_pp_and_optimized_curves
from helper import discover_plans_from_opti_csv


def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Create or retrieve a logger named 'pp_logger'. If debug=True, sets level=DEBUG; otherwise INFO.
    A stream handler is attached if none present.
    """
    logger = logging.getLogger("pp_logger")
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger


def run_pp_analysis(
    tradeplans: List[str],
    dataset_path: str,
    start_date: str,
    end_date: str,
    opti_csv_path: str,
    init_capital: float,
    top_n: int,
    debug: bool = False,
    concurrency: str = "process",
) -> None:
    """
    Orchestrates the Perfect Profit analysis by:
      1) Possibly auto-discovering tradeplans from opti_csv if tradeplans is empty.
      2) compute_pp_curves_for_dataset => writes 'pp_curves.csv'.
      3) compare_pp_and_optimized_curves => writes 'pp_correlation_*.csv'.

    :param tradeplans: A list of plan strings, e.g. ['1.0_5_1.5x_EMA2040']. 
                       If empty, we attempt discover_plans_from_opti_csv(opti_csv_path).
    :param dataset_path:  Path where dataset CSV files are stored.
    :param start_date:     e.g. "2025-01-01"
    :param end_date:       e.g. "2025-01-05"
    :param opti_csv_path:  CSV with columns like "2.0_5_1.5x_EMA520_91_61".
    :param init_capital:   Starting capital (float).
    :param top_n:          Number of top trades per day to sum for Perfect Profit.
    :param debug:          If True, enable debug logging.
    :param concurrency:    One of {"process","thread","sync"} for concurrency mode.
    """
    logger = setup_logger(debug)
    logger.info("Starting Perfect Profit analysis...")

    # If tradeplans is empty => discover them from opti_csv
    if not tradeplans:
        logger.info("No tradeplans provided => discovering from opti_csv columns...")
        discovered = discover_plans_from_opti_csv(opti_csv_path, debug=debug)
        if not discovered:
            raise ValueError(
                "No plan prefixes discovered from opti_csv. "
                "Cannot proceed with compute_pp_curves_for_dataset."
            )
        tradeplans = discovered
        logger.info("Using discovered tradeplans => %s", tradeplans)

    # Step 1: Compute PP curves => writes 'pp_curves.csv'
    pp_csv_path = "pp_curves.csv"
    compute_pp_curves_for_dataset(
        tradeplans=tradeplans,
        dataset_path=dataset_path,
        start_date=start_date,
        end_date=end_date,
        init_capital=init_capital,
        top_n=top_n,
        output_csv_path=pp_csv_path,
        debug=debug,
        concurrency=concurrency,
    )

    # Step 2: Compare PP curves with the optimized curves => correlation => writes CSV
    compare_pp_and_optimized_curves(
        pp_csv_path=pp_csv_path,
        opti_csv_path=opti_csv_path,
        start_date=start_date,
        end_date=end_date,
        debug=debug,
    )

    logger.info("Perfect Profit analysis completed.")


def main_cli() -> None:
    """
    Command-line entry point for Perfect Profit. 
    Parses sys.argv with argparse, then calls run_pp_analysis(...) 
    with the resulting arguments.
    """
    parser = argparse.ArgumentParser(description="CLI for Perfect Profit analyzer.")
    parser.add_argument("--dataset_path", required=True, help="Directory for dataset CSVs.")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--opti_csv_path", required=True, help="Path to the optimized CSV.")
    parser.add_argument("--init_capital", type=float, default=100000.0, help="Initial capital.")
    parser.add_argument("--top_n", type=int, default=2, help="Number of top trades per day.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--concurrency",
        choices=["process", "thread", "sync"],
        default="process",
        help="Which concurrency method to use.",
    )
    parser.add_argument(
        "--tradeplan",
        action="append",
        default=[],
        help=(
            "Specify a tradeplan prefix, e.g. '1.0_5_1.5x_EMA2040'. "
            "May be given multiple times. If empty => auto-discover from opti CSV."
        ),
    )

    args = parser.parse_args()

    run_pp_analysis(
        tradeplans=args.tradeplan,
        dataset_path=args.dataset_path,
        start_date=args.start_date,
        end_date=args.end_date,
        opti_csv_path=args.opti_csv_path,
        init_capital=args.init_capital,
        top_n=args.top_n,
        debug=args.debug,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main_cli()
