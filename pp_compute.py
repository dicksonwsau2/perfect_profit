"""
pp_compute.py

Holds logic for computing Perfect Profit curves (pp_curves.csv)
using multiple tradeplans. Formerly inside perfect_profit.py.
"""

import logging
import os
import pandas as pd
from typing import List
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)

from pp_worker import worker_pp_computation

def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Creates or retrieves a logger named 'pp_logger'.
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
    else:
        current_level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(current_level)
    return logger


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
    optionally applying 'BP adjustment'. Writes a combined CSV.

    1) For each plan => worker_pp_computation(...) => returns a Series of equity.
    2) If any plan fails => accumulate the errors => raise ValueError.
    3) Combine Series => write 'pp_curves.csv' (or whatever output_csv_path).

    :raises ValueError: if one or more tradeplans fail (like insufficient trades).
    """
    logger = setup_logger(debug)
    logger.info(
        "Computing PP curves for %d tradeplans with init_capital=%.2f, top_n=%d, bp_adjusted=%s...",
        len(tradeplans), init_capital, top_n, bp_adjusted
    )

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # concurrency => pick an Executor
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
        # synchronous
        for plan in tradeplans:
            try:
                curve_series = worker_pp_computation(
                    plan,
                    dataset_path,
                    date_range,
                    init_capital,
                    top_n,
                    bp_adjusted,
                    debug
                )
                results[plan] = curve_series
                logger.debug("Completed PP curve for tradeplan: %s", plan)
            except Exception as e:
                msg = f"Plan '{plan}' failed: {str(e)}"
                logger.error(msg)
                error_messages.append(msg)

    # If any errors occurred => raise
    if error_messages:
        combined = "\n".join(error_messages)
        raise ValueError(f"One or more tradeplans failed:\n{combined}")

    if not results:
        logger.warning("No PP curves were computed; output CSV not created.")
        return

    # Optionally check date index alignment
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

    # combine => final CSV
    pp_df = pd.DataFrame(results)
    pp_df.index.name = "Date"
    pp_df.reset_index(inplace=True)
    pp_df.to_csv(output_csv_path, index=False)
    logger.info("PP curves written to %s", output_csv_path)
