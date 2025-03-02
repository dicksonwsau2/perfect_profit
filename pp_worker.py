"""
pp_worker.py

This module contains the core logic and helper functions for Perfect Profit (PP)
analysis, including:

- Parsing tradeplans (e.g., "1.0_20-25-30_1.25x_EMA520")
- Reading relevant dataset files
- Computing daily Perfect Profit (PP)
- Generating an equity curve (with optional initial capital)
- The main `worker_pp_computation` function for orchestrating the above steps

All functions accept an optional `debug` parameter to enable local debug logging.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any
from datetime import datetime


def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Create or retrieve a logger named 'pp_logger'. If `debug` is True, set the
    logger level to DEBUG; otherwise INFO. A stream handler is attached if there
    are no existing handlers.

    :param debug: Whether to enable debug logs in this logger instance.
    :return: A configured logging.Logger object named 'pp_logger'.
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


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads the YAML config from config_path, returning a dict.
    Expects a structure like:
      tradeplan:
        MAX_WIDTH: 55
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_tradeplan(tradeplan: str, debug: bool = False) -> Dict[str, Any]:
    """
    Parse a tradeplan string into a parameter dictionary.

    The tradeplan format is assumed to be:
        "<premium>_<width>_<stop_loss>_<ema>"
    For example:
        "1.0_20-25-30_1.25x_EMA520"

    :param tradeplan: The tradeplan string to parse.
    :param debug: Enable debug-level logging for this function.
    :return: A dict with keys: 'premium', 'width', 'stop_loss', 'ema'.
    :raises ValueError: If the tradeplan does not have at least 4 segments.
    """
    logger = setup_logger(debug)
    logger.debug("parse_tradeplan called with tradeplan='%s'", tradeplan)

    parts = tradeplan.split('_')
    if len(parts) < 4:
        logger.error("Tradeplan '%s' does not have at least 4 parts.", tradeplan)
        raise ValueError(f"Invalid tradeplan format: '{tradeplan}'")

    parsed = {
        'premium': parts[0],
        'width': parts[1],
        'stop_loss': parts[2],
        'ema': parts[3]
    }
    logger.debug("Parsed tradeplan => %s", parsed)
    return parsed


def read_dataset_for_plan(
    dataset_path: str,
    plan_params: Dict[str, Any],
    debug: bool = False
) -> pd.DataFrame:
    """
    Locate and read exactly one CSV file matching the plan parameters in `dataset_path`,
    parsing the 'EntryTime' column as a datetime. 
    This accommodates datasets that do not have a separate 'date' column.

    The expected filename pattern is:
        "Dataset-Trades_{premium}_{width}_{stop_loss}_{ema}_*.csv"
    For example, if plan_params = {'premium':'1.0','width':'5','stop_loss':'1.5x','ema':'EMA2040'},
    we look for something like: "Dataset-Trades_1.0_5_1.5x_EMA2040_*.csv"

    In the matched CSV:
      - 'EntryTime' should be a column containing timestamps, e.g. '2023-01-03 09:33:00'
      - 'ProfitLossAfterSlippage' (and other columns) may be present.
      - We do NOT require a separate 'date' column because 'EntryTime' is parsed directly.

    Steps:
      1) Construct the filename pattern based on plan_params.
      2) Glob the pattern in dataset_path. Raise FileNotFoundError if 0 or >1 matches.
      3) Read the single matching CSV with parse_dates=['EntryTime'], ensuring it is not empty.

    :param dataset_path: 
        The directory containing CSV files named in the pattern 
        "Dataset-Trades_{premium}_{width}_{stop_loss}_{ema}_*.csv".
    :param plan_params: 
        Dictionary with keys: 'premium', 'width', 'stop_loss', 'ema'.
        Used to build the search filename pattern.
    :param debug: 
        If True, enable debug-level logging for this function.
    :return: 
        A pandas DataFrame parsed from the single matching CSV, with 'EntryTime' converted to datetime.
    :raises FileNotFoundError: 
        If no file matches the pattern, or more than one file matches, or the CSV is empty.
    """
    logger = setup_logger(debug)

    premium_str = plan_params['premium']
    width_str = plan_params['width']
    stoploss_str = plan_params['stop_loss']
    ema_str = plan_params['ema']

    filename_pattern = (
        f"Dataset-Trades_{premium_str}_{width_str}_{stoploss_str}_{ema_str}_*.csv"
    )
    pattern_path = os.path.join(dataset_path, filename_pattern)

    logger.debug("Looking for files matching pattern => %s", pattern_path)
    matched_files = glob.glob(pattern_path)
    logger.debug("Number of matched files: %d", len(matched_files))

    if len(matched_files) == 0:
        logger.error("No dataset files found for pattern: %s", pattern_path)
        raise FileNotFoundError(
            f"No matching dataset CSV for plan. Pattern: {pattern_path}"
        )
    if len(matched_files) > 1:
        logger.error("Multiple dataset files (%d) found for pattern: %s",
                     len(matched_files), pattern_path)
        raise FileNotFoundError(
            f"Multiple files for pattern {pattern_path}. Only one file expected."
        )

    file_path = matched_files[0]
    logger.debug("Reading dataset file => %s", file_path)

    # Parse 'EntryTime' as a datetime, matching real usage
    df = pd.read_csv(file_path, parse_dates=['EntryTime'])

    if df.empty:
        logger.error("DataFrame is empty after reading => %s", file_path)
        raise FileNotFoundError(
            f"Data read is empty for the matched file => {file_path}"
        )

    logger.debug("read_dataset_for_plan returning DataFrame with shape %s", df.shape)
    return df



def compute_daily_pp(
    df: pd.DataFrame,
    date_range: pd.DatetimeIndex,
    top_n: int,
    debug: bool = False
) -> pd.Series:
    """
    Compute daily Perfect Profit (PP) for each date in the dataset, within the 
    provided [start_day..end_day]. Each day must have >= top_n trades, or we raise ValueError.

    Steps:
      1) Convert 'EntryTime' to datetime if needed.
      2) Filter rows so only trades whose date is within [start_day..end_day].
      3) Group by 'TradeDate' and check if each group has >= top_n trades.
      4) Sort by 'ProfitLossAfterSlippage' descending, take top_n rows, sum them for PP.

    :param df: A DataFrame with columns like:
               - 'EntryTime' (string/datetime)
               - 'ProfitLossAfterSlippage' (float)
    :param date_range: A DatetimeIndex specifying the inclusive date window.
    :param top_n: The minimum number of trades per day. If a day's group has fewer than top_n, raise ValueError.
    :param debug: Enable debug-level logging.
    :return: A pd.Series indexed by day, named "PP". 
             Missing days are not included (only those that have trades).
    :raises ValueError: If the final filtered DataFrame is empty 
                        OR if any day has fewer than top_n trades.
    """
    logger = setup_logger(debug)
    logger.debug(
        "compute_daily_pp called with date_range=%s to %s, top_n=%d",
        date_range[0], date_range[-1], top_n
    )

    # Ensure 'EntryTime' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['EntryTime']):
        logger.debug("Converting 'EntryTime' to datetime.")
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])

    df['TradeDate'] = df['EntryTime'].dt.date

    start_day = date_range[0].date()
    end_day = date_range[-1].date()
    mask = (df['TradeDate'] >= start_day) & (df['TradeDate'] <= end_day)
    df = df[mask].copy()

    if df.empty:
        logger.error("No trades remain after date filter => [%s..%s]", start_day, end_day)
        raise ValueError(f"No trades remain after date filter => [{start_day}..{end_day}]")

    # Group by day
    grouped = df.groupby('TradeDate')
    daily_pp_list = []

    for day, group in grouped:
        # If the group has fewer than top_n trades, raise ValueError
        if len(group) < top_n:
            logger.error(
                "Day %s has only %d trades; need at least %d to compute PP",
                day, len(group), top_n
            )
            raise ValueError(
                f"Day {day} => only {len(group)} trades, need >= {top_n}"
            )

        group_sorted = group.sort_values('ProfitLossAfterSlippage', ascending=False)
        top_trades = group_sorted.head(top_n)
        day_pp = top_trades['ProfitLossAfterSlippage'].sum()
        daily_pp_list.append((day, day_pp))

    daily_pp_df = pd.DataFrame(daily_pp_list, columns=['TradeDate', 'PP'])
    daily_pp_df.sort_values('TradeDate', inplace=True)
    daily_pp_df.set_index('TradeDate', inplace=True)

    daily_pp_series = daily_pp_df['PP']
    daily_pp_series.name = "PP"

    logger.debug("compute_daily_pp returning Series with index: %s", daily_pp_series.index)
    return daily_pp_series


def create_equity_curve(
    daily_pp: pd.Series,
    init_capital: float,
    plan_params: Dict[str, Any],
    bp_adjusted: bool = False,
    debug: bool = False
) -> pd.Series:
    """
    Convert daily Perfect Profit values to a cumulative equity curve, starting at init_capital.

    :param daily_pp: A Series of daily PP, indexed by date.
    :param init_capital: The initial capital (float).
    :param plan_params: Dictionary containing tradeplan parameters like 'premium', 'width', etc.
    :param bp_adjusted: If True, adjusts the daily PP based on a scaling factor.
    :param debug: Enable debug-level logging.
    :return: A Series of the same index representing the equity curve: init_capital + cumsum(daily_pp).
    """
    logger = setup_logger(debug)
    logger.debug(
        "create_equity_curve called with init_capital=%.2f, daily_pp length=%d",
        init_capital, len(daily_pp)
    )

    if bp_adjusted:
        # Get scaling factor from configuration and apply it to daily_pp
        config = load_config()
        max_width = config['tradeplan']['MAX_WIDTH']
        scaling_factor = max_width / float(plan_params['width'])
        logger.debug(f"Scaling factor applied: {scaling_factor}")
        daily_pp = daily_pp * scaling_factor

    # Calculate the cumulative sum of daily PP and add the initial capital
    equity_curve = init_capital + daily_pp.cumsum()

    # Attach the plan_params dictionary directly to the equity curve
    equity_curve.attrs['plan_params'] = plan_params

    logger.debug("create_equity_curve returning final equity curve.")
    return equity_curve


def worker_pp_computation(
    tradeplan: str,
    dataset_path: str,
    date_range: pd.DatetimeIndex,
    init_capital: float,
    top_n: int,
    bp_adjusted: bool = True,  # Defaulting to True
    debug: bool = False
) -> pd.Series:
    """
    Orchestrate a single tradeplan's Perfect Profit computation.

    Steps:
      1) parse_tradeplan
      2) read_dataset_for_plan
      3) compute_daily_pp (using top_n trades)
      4) create_equity_curve using init_capital and bp_adjusted
      5) Return the resulting equity curve with plan_params in equity_curve.attrs

    :param tradeplan: The string describing the plan (e.g. "1.0_5_1.5x_EMA2040").
    :param dataset_path: Folder containing relevant CSV dataset(s).
    :param date_range: A range of dates (Timestamp-based) for the analysis.
    :param init_capital: Initial capital to offset or scale the final equity.
    :param top_n: Number of top trades per day to sum for Perfect Profit.
    :param bp_adjusted: If True, scale the daily PP based on MAX_WIDTH and plan width.
    :param debug: Enable debug-level logging.
    :return: A pandas Series (equity curve). The plan_params are stored in equity_curve.attrs['plan_params'].
    """
    logger = setup_logger(debug)
    logger.debug(
        "worker_pp_computation started for tradeplan=%s, dataset_path=%s, "
        "init_capital=%.2f, top_n=%d, bp_adjusted=%s",
        tradeplan, dataset_path, init_capital, top_n, bp_adjusted
    )

    # Step 1: Parse the tradeplan string into plan parameters (premium, width, stop_loss, ema)
    plan_params = parse_tradeplan(tradeplan, debug=debug)

    # Step 2: Read the dataset associated with the tradeplan
    df_data = read_dataset_for_plan(dataset_path, plan_params, debug=debug)

    # Step 3: Compute the daily PP for the tradeplan's dataset
    daily_pp_series = compute_daily_pp(df_data, date_range, top_n=top_n, debug=debug)

    # Step 4: Create the equity curve (with or without BP adjustment)
    equity_curve = create_equity_curve(
        daily_pp_series,
        init_capital,
        plan_params,
        bp_adjusted,
        debug=debug
    )

    logger.debug("worker_pp_computation completed for tradeplan=%s", tradeplan)
    return equity_curve
