"""
perfect_profit.py

Main orchestrator for Perfect Profit analysis, with optional concurrency and usage of a dataset_path.
"""

import logging
from typing import List

from correlation_manager import compare_pp_and_optimized_curves
from pp_compute import compute_pp_curves_for_dataset  # <--- now we import from new module

# if you have a helper function to discover tradeplans from the opti_csv
from helper import discover_plans_from_opti_csv  # if not empty => skip

def setup_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger('pp_logger')
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        current_level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(current_level)
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
    concurrency: str = "process"
) -> None:
    """
    Orchestrates the Perfect Profit analysis by:
      1) If tradeplans is empty => discover from opti_csv using discover_plans_from_opti_csv.
      2) compute_pp_curves_for_dataset => writes pp_curves.csv
      3) compare_pp_and_optimized_curves => writes correlation CSV.
    """
    logger = setup_logger(debug)
    logger.info("Starting Perfect Profit analysis...")

    if not tradeplans:
        logger.info("No tradeplans provided => discovering from opti_csv columns...")
        discovered = discover_plans_from_opti_csv(opti_csv_path, debug=debug)
        if not discovered:
            raise ValueError(
                "No plan prefixes discovered from opti_csv. "
                "Cannot proceed with compute_pp_curves_for_dataset."
            )
        tradeplans = discovered
        logger.info(f"Using discovered tradeplans => {tradeplans}")

    pp_csv_path = "pp_curves.csv"

    # step 2 => compute pp curves => final => "pp_curves.csv"
    compute_pp_curves_for_dataset(
        tradeplans=tradeplans,
        dataset_path=dataset_path,
        start_date=start_date,
        end_date=end_date,
        init_capital=init_capital,
        top_n=top_n,
        output_csv_path=pp_csv_path,
        debug=debug,
        concurrency=concurrency
    )

    # step 3 => correlation => writes "pp_correlation_yyyyMMdd_HHMMSS.csv"
    compare_pp_and_optimized_curves(
        pp_csv_path=pp_csv_path,
        opti_csv_path=opti_csv_path,
        start_date=start_date,
        end_date=end_date,
        debug=debug
    )

    logger.info("Perfect Profit analysis completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI for Perfect Profit analyzer.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--opti_csv_path", required=True)
    parser.add_argument("--init_capital", type=float, default=100000.0)
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--concurrency", choices=["process","thread","sync"], default="process")

    # user can skip => auto-discover from opti, or pass them => override
    parser.add_argument("--tradeplan", action="append", default=[],
                        help="Add plan prefix, e.g. '1.0_5_1.5x_EMA2040'. If empty => auto-discover from opti CSV.")

    args = parser.parse_args()

    run_pp_analysis(
        tradeplans=args.tradeplan,  # might be empty => auto-discover from opti
        dataset_path=args.dataset_path,
        start_date=args.start_date,
        end_date=args.end_date,
        opti_csv_path=args.opti_csv_path,
        init_capital=args.init_capital,
        top_n=args.top_n,
        debug=args.debug,
        concurrency=args.concurrency
    )
