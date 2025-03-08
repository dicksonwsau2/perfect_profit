# correlation_manager.py

import os
import hashlib
import logging
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from typing import Tuple

def setup_logger(debug: bool = False) -> logging.Logger:
    """
    Creates or retrieves a logger named 'pp_logger'. 
    If debug=True, sets level=DEBUG, otherwise INFO.
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


def load_and_filter_data(
    csv_path: str,
    start_date: str,
    end_date: str,
    debug: bool
) -> pd.DataFrame:
    """
    Loads a CSV file from 'csv_path' where the first column is named 'Date'.
    The code uses 'index_col="Date"', parse_dates=True to treat that column 
    as a datetime index. Then we filter rows to [start_date..end_date].

    Example:
        Suppose your CSV looks like:

        Date,3.0_5_1.5x_EMA520
        2025-01-01,100010
        2025-01-02,100030
        2025-01-03,100050

        Then we do:
            df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            => df.index => DatetimeIndex(["2025-01-01", ...], dtype='datetime64[ns]', name='Date')
            => we then filter [start_date..end_date].

    :param csv_path:   The path to the CSV file containing a 'Date' column.
    :param start_date: Lower bound (YYYY-MM-DD).
    :param end_date:   Upper bound (YYYY-MM-DD).
    :param debug:      If True, enable debug logging.
    :return:           A filtered and sorted DataFrame indexed by 'Date'.
    :raises FileNotFoundError: If the file doesn't exist.
    """
    logger = setup_logger(debug)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found => {csv_path}")

    # Here we read with index_col="Date", matching the capital "D" in the CSV.
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    df.sort_index(inplace=True)

    return df


def verify_date_coverage(
    pp_df: pd.DataFrame,
    opti_df: pd.DataFrame,
    debug: bool = False
) -> None:
    """
    Ensures pp_df and opti_df share an identical set of dates after filtering.

    If either set of dates is missing from the other, or the lengths differ,
    we raise ValueError.

    :param pp_df:   The DataFrame of Perfect Profit curves (indexed by 'Date').
    :param opti_df: The DataFrame of optimized curves (also indexed by 'Date').
    :param debug:   If True, enable debug logging.
    """
    logger = setup_logger(debug)

    dates_pp = set(pp_df.index)
    dates_opti = set(opti_df.index)
    if dates_pp != dates_opti:
        missing_in_opti = dates_pp - dates_opti
        missing_in_pp   = dates_opti - dates_pp
        msg = (
            "Mismatched date coverage:\n"
            f"Dates in pp_df not in opti_df => {sorted(missing_in_opti)}\n"
            f"Dates in opti_df not in pp_df => {sorted(missing_in_pp)}"
        )
        logger.error(msg)
        raise ValueError(msg)

    # Double-check row counts too
    if len(pp_df) != len(opti_df):
        msg = (
            f"Row mismatch => pp_df has {len(pp_df)} rows, "
            f"opti_df has {len(opti_df)} rows."
        )
        logger.error(msg)
        raise ValueError(msg)


def parse_opti_column(
    opti_col: str,
    debug: bool = False
) -> Tuple[str, str, str, str, str, str]:
    """
    Splits an optimized column name of the form:
      "<premium>_<width>_<SL>_<EMA>_<OPL>_<EPL>"
    Example: "3.14_5_1.5x_EMA520_91_61"
      => 6 parts => first => premium => parse float => 1 decimal => "3.1"

    :param opti_col: e.g. "3.14_5_1.5x_EMA520_91_61"
    :param debug:    If True, debug logging
    :return:         (premium_str, width, SL, EMA, OPL, EPL) 
                     e.g. ("3.1","5","1.5x","EMA520","91","61")
    :raises ValueError: if not EXACTLY 6 parts, or premium fails to parse.
    """
    logger = setup_logger(debug)

    parts = opti_col.split("_")
    if len(parts) != 6:
        msg = f"Opti col '{opti_col}' must have EXACTLY 6 parts; got {len(parts)}."
        logger.error(msg)
        raise ValueError(msg)

    raw_premium, width, SL, EMA, OPL, EPL = parts

    try:
        pval = float(raw_premium)
        premium_str = f"{pval:.1f}"  # enforce 1 decimal
    except ValueError as e:
        msg = f"Failed to parse '{raw_premium}' as float in col '{opti_col}'."
        logger.error(msg)
        raise ValueError(msg) from e

    return (premium_str, width, SL, EMA, OPL, EPL)


def compare_pp_and_optimized_curves(
    pp_csv_path: str,
    opti_csv_path: str,
    start_date: str,
    end_date: str,
    debug: bool = False
) -> pd.DataFrame:
    """
    High-level correlation function that:
      1) Loads 'pp_csv_path' and 'opti_csv_path' with 'Date' as index_col.
      2) Filters each to [start_date..end_date].
      3) verify_date_coverage(...) => ensure same coverage.
      4) For each column in 'opti_df':
           parse_opti_column(...) => (premium_str, width, SL, EMA, OPL, EPL)
           => plan_prefix = f"{premium_str}_{width}_{SL}_{EMA}"
           => correlation => final day => build MD5 opti_id
      5) Return a DataFrame with columns:
           [
             "opti_id","premium","width","SL","EMA","OPL","EPL",
             "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
           ]
         also written to "pp_correlation_<timestamp>.csv".

    Example CSV snippet for pp_csv_path:

      Date,3.0_5_1.5x_EMA520
      2025-01-01,100010
      2025-01-02,100030
      2025-01-03,100050

    Example CSV snippet for opti_csv_path:

      Date,3.0_5_1.5x_EMA520_91_61
      2025-01-01,200020
      2025-01-02,200040
      2025-01-03,200060

    The code reads each with:
      pd.read_csv(..., index_col="Date", parse_dates=True)

    => we do pearsonr(...) on each matched pair.

    :param pp_csv_path:   The path to a "pp_curves.csv", 
                          first column heading "Date", 
                          subsequent columns like "3.0_5_1.5x_EMA520".
    :param opti_csv_path: The path to a "opti_curves.csv", 
                          first column "Date", 
                          subsequent columns like "3.0_5_1.5x_EMA520_91_61".
    :param start_date:    e.g. "2025-01-01"
    :param end_date:      e.g. "2025-01-31"
    :param debug:         If True, debug logging.
    :return: A DataFrame with columns:
             [
               "opti_id","premium","width","SL","EMA","OPL","EPL",
               "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
             ]
    """
    logger = setup_logger(debug)
    logger.info(
        "Comparing PP curves in '%s' with optimized curves in '%s' using strict coverage.",
        pp_csv_path, opti_csv_path
    )

    # 1) Load & filter
    pp_df   = load_and_filter_data(pp_csv_path,   start_date, end_date, debug)
    opti_df = load_and_filter_data(opti_csv_path, start_date, end_date, debug)
    
    # 2) IMMEDIATE NO-DATA CHECKS
    if len(pp_df) == 0:
        msg = "No data remain in PP CSV after date filtering."
        logger.error(msg)
        raise ValueError(msg)

    if len(opti_df) == 0:
        msg = "No data remain in OPTI CSV after date filtering."
        logger.error(msg)
        raise ValueError(msg)

    # 3) Verify coverage
    verify_date_coverage(pp_df, opti_df, debug)

    results = []

    # 4) For each col in opti_df => parse => correlation
    from scipy.stats import pearsonr

    for opti_col in opti_df.columns:
        premium_str, width, SL, EMA, OPL, EPL = parse_opti_column(opti_col, debug)
        plan_prefix = f"{premium_str}_{width}_{SL}_{EMA}"

        if plan_prefix not in pp_df.columns:
            msg = (
                f"Optimized col '{opti_col}' => plan_prefix '{plan_prefix}' "
                f"not found in pp_df => {list(pp_df.columns)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        series_opti = opti_df[opti_col]
        series_pp   = pp_df[plan_prefix]

        if len(series_opti) != len(series_pp):
            msg = (
                f"Series length mismatch => col='{opti_col}', prefix='{plan_prefix}' => "
                f"{len(series_opti)} vs {len(series_pp)}"
            )
            logger.error(msg)
            raise ValueError(msg)

        if series_opti.empty:
            msg = f"No data for col='{opti_col}', plan_prefix='{plan_prefix}'"
            logger.error(msg)
            raise ValueError(msg)

        corr_val, p_val = pearsonr(series_pp, series_opti)

        # final day
        opti_pl = float(series_opti.iloc[-1])
        pp_pl   = float(series_pp.iloc[-1])

        # build final_str => MD5 => opti_id
        final_str = f"{premium_str}_{width}_{SL}_{EMA}_{OPL}_{EPL}"
        opti_id   = hashlib.md5(final_str.encode('utf-8')).hexdigest()

        results.append((
            opti_id,
            premium_str,
            width, SL, EMA, OPL, EPL,
            corr_val,
            p_val,
            opti_pl,
            pp_pl
        ))

    if not results:
        msg = "No columns matched or no correlation computed."
        logger.warning(msg)
        return pd.DataFrame(columns=[
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ])

    # Build final DataFrame => write & return
    corr_df = pd.DataFrame(
        results,
        columns=[
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ]
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_name = f"pp_correlation_{timestamp}.csv"
    corr_df.to_csv(csv_name, index=False)
    logger.info("Wrote correlation metrics to %s", csv_name)

    return corr_df
