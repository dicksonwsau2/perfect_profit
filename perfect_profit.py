"""
perfect_profit.py

Implements Perfect Profit (PP) analysis with optional concurrency for computing
multiple PP curves in parallel, plus usage of a dataset_path for reading CSV data.

Main Workflow:
1) run_pp_analysis:
   - Accepts tradeplans, dataset_path, date range, init_capital, top_n, etc.
   - Calls compute_pp_curves_for_dataset to create PP curves CSV.
   - Calls compare_pp_and_optimized_curves to produce correlation metrics.

2) compute_pp_curves_for_dataset:
   - Takes tradeplans, dataset_path, date range, init_capital, top_n, concurrency mode.
   - Spawns worker computations (worker_pp_computation) to produce each tradeplan’s PP curve.
   - Writes a CSV of all combined PP curves.

3) worker_pp_computation:
   - Receives a single tradeplan, dataset_path, date_range, init_capital, top_n, etc.
   - Possibly read data from dataset_path and compute daily PP.

4) compare_pp_and_optimized_curves:
   - Reads the PP CSV, compares it to an optimized CSV, computes Pearson correlation.

5) pearson_correlation:
   - Calculates Pearson correlation between two Series, returning 0.0 if empty.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import List
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)
from scipy.stats import pearsonr
from pp_worker import worker_pp_computation
from correlation_manager import compare_pp_and_optimized_curves


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
    opti_curves_path: str,
    init_capital: float,
    top_n: int,
    debug: bool = False,
    concurrency: str = "process"
) -> None:
    """
    Orchestrates the Perfect Profit analysis:
      1) Compute PP curves for the given tradeplans over [start_date..end_date],
         using dataset_path if needed for reading data, with init_capital and top_n.
      2) Compare those PP curves to optimized curves, producing correlation metrics.

    :param tradeplans:       A list of tradeplan identifiers (e.g. ["1.0_5_1.5x_EMA2040", ...]).
    :param dataset_path:     The filesystem path where dataset CSVs are stored.
    :param start_date:       The start date of the analysis window (YYYY-MM-DD).
    :param end_date:         The end date of the analysis window (YYYY-MM-DD).
    :param opti_curves_path: File path to a CSV of optimized curves (or directory if logic differs).
    :param init_capital:     The initial capital for the strategy (float).
    :param top_n:            How many top trades per day to sum for Perfect Profit.
    :param debug:            Whether to enable detailed debug logging.
    :param concurrency:      Concurrency mode for computing PP curves ("process", "thread", or "sync").
    """
    logger = setup_logger(debug)
    logger.info("Starting Perfect Profit analysis...")

    # Output file for PP curves
    pp_csv_path = "pp_curves.csv"

    # Step 1: Compute PP curves
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

    # Step 2: Compare PP curves with optimized curves
    compare_pp_and_optimized_curves(
        pp_csv_path=pp_csv_path,
        opti_curves_path=opti_curves_path,
        start_date=start_date,
        end_date=end_date,
        debug=debug
    )

    logger.info("Perfect Profit analysis completed.")


def compute_pp_curves_for_dataset(
    tradeplans: List[str],
    dataset_path: str,
    start_date: str,
    end_date: str,
    init_capital: float,
    top_n: int,
    output_csv_path: str,
    bp_adjusted: bool = True,
    concurrency: str = "process",
    debug: bool = False
) -> None:
    """
    Computes Perfect Profit (PP) curves for each tradeplan in the given date window,
    optionally applying 'BP adjustment' if bp_adjusted=True.
    If any plan fails, we accumulate the original error messages and raise a combined ValueError.

    Steps:
      1) For each tradeplan, run worker_pp_computation(...) in parallel (or sync).
      2) If any plan fails, store the exception message for that plan in a list.
      3) At the end, if errors occurred, raise ValueError with all the original messages appended.
      4) If no errors, combine the resulting Series into a DataFrame => CSV.

    :raises ValueError:
        If any plan's worker_pp_computation fails (like insufficient trades).
        We include the original messages so tests can match e.g. "only 1 trades, need >= 2".
    """
    logger = setup_logger(debug)
    logger.info(
        "Computing PP curves for %d tradeplans with init_capital=%.2f, top_n=%d, bp_adjusted=%s...",
        len(tradeplans), init_capital, top_n, bp_adjusted
    )

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    if concurrency == "process":
        ExecutorClass = ProcessPoolExecutor
    elif concurrency == "thread":
        ExecutorClass = ThreadPoolExecutor
    elif concurrency == "sync":
        ExecutorClass = None
    else:
        logger.warning("Unknown concurrency mode '%s', defaulting to 'process'.", concurrency)
        ExecutorClass = ProcessPoolExecutor

    results = {}
    # We'll store error messages from any failing plan
    error_messages = []

    if ExecutorClass is not None:
        with ExecutorClass() as executor:
            futures = {}
            for plan in tradeplans:
                futures[executor.submit(
                    worker_pp_computation,
                    plan,
                    dataset_path,
                    date_range,
                    init_capital,
                    top_n,
                    bp_adjusted,
                    debug
                )] = plan

            for future in as_completed(futures):
                plan_name = futures[future]
                try:
                    curve_series = future.result()
                    results[plan_name] = curve_series
                    logger.debug("Completed PP curve for tradeplan: %s", plan_name)
                except Exception as e:
                    msg = f"Plan '{plan_name}' failed: {str(e)}"
                    logger.error(msg)
                    error_messages.append(msg)
    else:
        # Synchronous path
        for plan in tradeplans:
            try:
                curve_series = worker_pp_computation(
                    plan, dataset_path, date_range, init_capital,
                    top_n, bp_adjusted, debug
                )
                results[plan] = curve_series
                logger.debug("Completed PP curve for tradeplan: %s", plan)
            except Exception as e:
                msg = f"Plan '{plan}' failed: {str(e)}"
                logger.error(msg)
                error_messages.append(msg)

    # If we have any errors, raise a combined ValueError
    if error_messages:
        combined = "\n".join(error_messages)
        raise ValueError(f"One or more tradeplans failed:\n{combined}")

    if not results:
        logger.warning("No PP curves were computed; output CSV not created.")
        return

    # (Optional) check date index alignment
    plan_items = list(results.items())
    first_plan, first_series = plan_items[0]
    for plan_name, curve_series in plan_items[1:]:
        if not curve_series.index.equals(first_series.index):
            mismatch_msg = (
                f"Date index mismatch: '{plan_name}' differs from '{first_plan}'. "
                "Aborting CSV creation."
            )
            logger.error(mismatch_msg)
            raise ValueError(mismatch_msg)

    # Combine columns => final CSV
    pp_df = pd.DataFrame(results)
    pp_df.index.name = "Date"
    pp_df.reset_index(inplace=True)
    pp_df.to_csv(output_csv_path, index=False)
    logger.info("PP curves written to %s", output_csv_path)