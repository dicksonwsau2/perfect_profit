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


def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Configure and return a logger instance for debug or info messages.

    :param debug: If True, sets logger level to DEBUG, else INFO.
    :return: A logging.Logger object named 'pp_logger'.
    """
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



def compare_pp_and_optimized_curves(
    pp_csv_path: str,
    opti_curves_path: str,
    start_date: str,
    end_date: str,
    debug: bool = False
) -> None:
    """
    Reads the PP curves from `pp_csv_path` and compares each column (tradeplan)
    to a matching column in the optimized curves file at `opti_curves_path`.
    Filters both by [start_date..end_date] and computes Pearson correlation.

    Writes a CSV named "pp_correlation.csv" containing columns:
      - Tradeplan
      - PearsonCorrelation

    :param pp_csv_path:     Path to the CSV of PP curves.
    :param opti_curves_path: Path to a CSV of optimized curves (columns should match tradeplan names).
    :param start_date:       Start date for the comparison window (YYYY-MM-DD).
    :param end_date:         End date for the comparison window (YYYY-MM-DD).
    :param debug:            Whether to enable debug-level logging.
    """
    logger = setup_logger(debug)
    logger.info(
        "Comparing PP curves in %s with optimized curves in %s.",
        pp_csv_path, opti_curves_path
    )

    # Load the PP curves
    try:
        pp_df = pd.read_csv(pp_csv_path, index_col='date', parse_dates=True)
    except FileNotFoundError:
        logger.error("Could not find PP CSV at %s", pp_csv_path)
        return

    # Load the optimized curves
    try:
        opti_df = pd.read_csv(opti_curves_path, index_col='date', parse_dates=True)
    except FileNotFoundError:
        logger.error("Could not find optimized curves at %s", opti_curves_path)
        return

    # Filter by date range
    mask_pp = (pp_df.index >= start_date) & (pp_df.index <= end_date)
    mask_opti = (opti_df.index >= start_date) & (opti_df.index <= end_date)
    pp_df = pp_df.loc[mask_pp]
    opti_df = opti_df.loc[mask_opti]

    correlations = []
    for plan in pp_df.columns:
        if plan in opti_df.columns:
            corr_val = pearson_correlation(pp_df[plan], opti_df[plan], debug)
            correlations.append((plan, corr_val))
        else:
            logger.warning("No matching optimized column found for '%s'", plan)

    if correlations:
        corr_df = pd.DataFrame(correlations, columns=['Tradeplan', 'PearsonCorrelation'])
        corr_df.to_csv("pp_correlation.csv", index=False)
        logger.info("Wrote correlation metrics to pp_correlation.csv")
    else:
        logger.warning("No correlations computed. Possibly no matching tradeplans.")


def pearson_correlation(series_a: pd.Series, series_b: pd.Series, debug: bool = False) -> float:
    """
    Computes the Pearson Correlation coefficient between two pd.Series.

    :param series_a: First time series.
    :param series_b: Second time series.
    :param debug:    If True, uses debug-level logging.
    :return:         The Pearson correlation (float). Returns 0.0 if either series is empty.
    """
    logger = setup_logger(debug)
    if series_a.empty or series_b.empty:
        logger.debug("One or both series are empty; returning correlation=0.0.")
        return 0.0

    from scipy.stats import pearsonr
    min_len = min(len(series_a), len(series_b))
    slice_a = series_a.iloc[:min_len]
    slice_b = series_b.iloc[:min_len]

    corr_val, _ = pearsonr(slice_a, slice_b)
    return corr_val


if __name__ == "__main__":
    """
    Example usage of run_pp_analysis with placeholder arguments.
    Customize these values or replace with argument parsing (e.g. run_cli).
    """
    sample_tradeplans = ["1.0_5_1.5x_EMA2040", "1.5_10_2.0x_EMA540"]
    sample_dataset_path = "path/to/your/dataset"
    sample_start_date = "2025-01-01"
    sample_end_date = "2025-01-05"
    sample_opti_path = "optimized_curves.csv"
    sample_init_capital = 100_000
    sample_top_n = 10  # new argument

    run_pp_analysis(
        tradeplans=sample_tradeplans,
        dataset_path=r".\sample_dataset.csv",
        start_date=sample_start_date,
        end_date=sample_end_date,
        opti_curves_path=sample_opti_path,
        init_capital=sample_init_capital,
        top_n=sample_top_n,
        debug=True,
        concurrency="thread"
    )
