"""
helper.py

Contains helper functions for the Perfect Profit analyzer,
such as discovering tradeplan prefixes from an optimized CSV.
"""

import os
import logging
import pandas as pd

def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Creates or retrieves a logger named 'pp_logger'.
    If debug=True, set level=DEBUG, otherwise INFO.
    A stream handler is attached if none present.
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
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger

def discover_plans_from_opti_csv(opti_csv_path: str, debug: bool = False) -> list:
    """
    Reads the columns from opti_csv_path, ignoring 'Date' if present.
    For each column like '3.0_5_1.5x_EMA520_91_61', parse out
    the prefix '3.0_5_1.5x_EMA520' (the first 4 segments).
    Returns a sorted list of unique prefixes.

    Example usage:
      => discover_plans_from_opti_csv("opti_curves.csv")
      => ["3.0_5_1.5x_EMA520", "4.0_10_2.0x_EMA540", ...]

    :param opti_csv_path: Path to the 'opti' CSV (with columns e.g. "3.0_5_1.5x_EMA520_91_61").
    :param debug: If True, enable debug logging.
    :return: A list of plan prefix strings (like "3.0_5_1.5x_EMA520").
    """
    logger = setup_logger(debug)

    if not os.path.exists(opti_csv_path):
        raise FileNotFoundError(f"Opti CSV not found => {opti_csv_path}")

    df_opti = pd.read_csv(opti_csv_path)
    if "Date" in df_opti.columns:
        # remove the 'Date' column to process only the plan columns
        df_opti.drop(columns=["Date"], inplace=True, errors="ignore")

    plan_prefixes = set()
    for col in df_opti.columns:
        # e.g. col = "3.0_5_1.5x_EMA520_91_61"
        parts = col.split("_")
        if len(parts) < 4:
            logger.warning(f"Skipping column '{col}' => not enough segments for a plan prefix.")
            continue
        # first 4 segments => "3.0_5_1.5x_EMA520"
        prefix = "_".join(parts[:4])
        plan_prefixes.add(prefix)

    discovered = sorted(plan_prefixes)
    logger.info(f"Discovered {len(discovered)} plan prefixes from {opti_csv_path}: {discovered}")
    return discovered
