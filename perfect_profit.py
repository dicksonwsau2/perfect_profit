import os
import pandas as pd
from typing import List
from scipy.stats import pearsonr
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def perform_pp_analysis(
    accumulated_pnl_paths: List[str],
    evl_start_fmt: str,
    dataset_end_fmt: str,
    timestamp: str,
    summary_dir: str,
    initial_capital: float,
    compression: str = 'snappy'
) -> None:
    """
    Orchestrates the Perfect Profit (PP) analysis:
    1) Generate PP equity curves.
    2) Compare PP curves with optimized curves and calculate Pearson Correlation.
    3) Store the results.
    """
    try:
        # Step 1: Generate PP curves
        pp_curves = generate_pp_curves(accumulated_pnl_paths, evl_start_fmt, dataset_end_fmt, initial_capital)
        
        # Step 2: Compare PP and Optimized curves (Pearson Correlation calculation)
        correlation_results = compare_pp_and_optimized_curves(pp_curves, summary_dir)
        
        # Step 3: Store or output results
        save_pp_analysis_results(pp_curves, correlation_results, summary_dir)
        logger.info("PP analysis completed successfully.")
    except Exception as e:
        logger.error(f"Failed during PP analysis: {e}")
        raise e


def generate_pp_curves(
    accumulated_pnl_paths: List[str],
    evl_start_fmt: str,
    dataset_end_fmt: str,
    initial_capital: float
) -> List[pd.DataFrame]:
    """
    Generates PP equity curves for the top N most profitable trade plans.
    """
    pp_curves = []
    for file_path in accumulated_pnl_paths:
        # Read the PnL data
        pnl_data = pd.read_csv(file_path)

        # Filter and process the data (select top N profitable entry times)
        pp_curve = process_pp_data(pnl_data, evl_start_fmt, dataset_end_fmt, initial_capital)
        pp_curves.append(pp_curve)
    
    return pp_curves


def process_pp_data(
    pnl_data: pd.DataFrame,
    evl_start_fmt: str,
    dataset_end_fmt: str,
    initial_capital: float
) -> pd.DataFrame:
    """
    Process the data to select top N profitable entry times and calculate daily PnL.
    """
    # Filter the data based on start and end dates
    pnl_data['date'] = pd.to_datetime(pnl_data['date'])
    pnl_data = pnl_data[(pnl_data['date'] >= evl_start_fmt) & (pnl_data['date'] <= dataset_end_fmt)]

    # Select top N profitable entry times each day (for simplicity, take top 3)
    daily_pnl = pnl_data.groupby('date').apply(
        lambda x: x.nlargest(3, 'profit')['profit'].sum()
    )

    # Create PP equity curve
    pp_curve = daily_pnl.cumsum() + initial_capital  # Accumulated PnL
    return pp_curve


def compare_pp_and_optimized_curves(
    pp_curves: List[pd.DataFrame],
    summary_dir: str
) -> List[float]:
    """
    Compare PP curves with optimized curves and calculate Pearson Correlation coefficient.
    """
    correlation_results = []
    for pp_curve in pp_curves:
        # Load the corresponding optimized curve (assumed to be in the summary_dir)
        optimized_curve = load_optimized_curve(summary_dir)
        
        # Calculate Pearson Correlation coefficient
        correlation = pearson_correlation(pp_curve, optimized_curve)
        correlation_results.append(correlation)
    
    return correlation_results


def load_optimized_curve(summary_dir: str) -> pd.DataFrame:
    """
    Load the optimized equity curve from the summary directory.
    """
    # This is just an example, modify as needed to locate the optimized curve.
    optimized_curve_path = os.path.join(summary_dir, "optimized_curve.csv")
    return pd.read_csv(optimized_curve_path)


def pearson_correlation(pp_curve: pd.DataFrame, optimized_curve: pd.DataFrame) -> float:
    """
    Calculate Pearson Correlation coefficient between PP and optimized curve.
    """
    return pearsonr(pp_curve, optimized_curve)[0]


def save_pp_analysis_results(
    pp_curves: List[pd.DataFrame],
    correlation_results: List[float],
    summary_dir: str
) -> None:
    """
    Save PP curves and Pearson correlation results to the summary directory.
    """
    # Save PP curves to summary directory
    for idx, pp_curve in enumerate(pp_curves):
        pp_curve.to_csv(f"{summary_dir}/pp_curve_{idx}.csv", compression='snappy')
    
    # Save Pearson correlation results
    correlation_df = pd.DataFrame(correlation_results, columns=["Pearson Correlation"])
    correlation_df.to_csv(f"{summary_dir}/pp_correlation.csv", index=False, compression='snappy')

    logger.info(f"Saved PP analysis results to {summary_dir}")


# Optionally: You may add other utility functions as needed, like loading PnL data,
# processing dates, or handling edge cases.
