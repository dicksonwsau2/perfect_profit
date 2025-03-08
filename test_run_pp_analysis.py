"""
test_run_pp_analysis.py

Integration test for run_pp_analysis(...) in perfect_profit.py.
Ensures the entire pipeline runs end-to-end, producing pp_curves.csv
and pp_correlation_*.csv in a temporary environment.
"""

import unittest
import os
import tempfile
import glob
import pandas as pd
from datetime import datetime
from perfect_profit import run_pp_analysis

class TestRunPPAnalysis(unittest.TestCase):
    """
    Integration test for run_pp_analysis.
    We'll create a minimal dataset CSV in dataset_path and a minimal
    optimized CSV, then let run_pp_analysis do its work, ensuring
    config.yaml is also present in the ephemeral directory.
    """

    def setUp(self):
        """
        Create a temporary directory as dataset_path, plus an empty
        directory or file for opti_curves_path. We'll also switch our
        working directory to it so that the final 'pp_correlation_*.csv'
        ends up ephemeral as well.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        # 1) Create a minimal config.yaml
        config_yaml_path = os.path.join(self.temp_dir.name, "config.yaml")
        with open(config_yaml_path, "w") as f:
            f.write("tradeplan:\n  MAX_WIDTH: 55\n")

        # 2) Create the "datasets" folder where the CSV for the plan will live
        self.dataset_path = os.path.join(self.temp_dir.name, "datasets")
        os.makedirs(self.dataset_path, exist_ok=True)

        # We'll create a single plan => "1.0_5_1.5x_EMA2040"
        self.plan_str = "1.0_5_1.5x_EMA2040"
        dataset_csv_name = f"Dataset-Trades_{self.plan_str}_test.csv"
        self.dataset_csv_path = os.path.join(self.dataset_path, dataset_csv_name)

        # Write a minimal dataset CSV with 'EntryTime' and
        # 'ProfitLossAfterSlippage' covering 3 days
        data_rows = []
        base_date = datetime(2025, 1, 1, 9, 0)
        # We'll do 2 trades per day for 3 days => total 6 rows
        for day_offset in range(3):
            for minute_offset in [0, 15]:
                entry_dt = base_date.replace(hour=9, minute=minute_offset) + \
                           pd.Timedelta(days=day_offset)
                profit = 10.0 + day_offset*2 + minute_offset*0.1
                data_rows.append({
                    "EntryTime": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "ProfitLossAfterSlippage": profit
                })
        df_dataset = pd.DataFrame(data_rows)
        df_dataset.to_csv(self.dataset_csv_path, index=False)

        # 3) Create a minimal optimized CSV => 3 days => "1.0_5_1.5x_EMA2040_91_61"
        self.opti_curves_path = os.path.join(self.temp_dir.name, "opti_curves.csv")
        idx = pd.date_range("2025-01-01", periods=3, freq='D')
        df_opti = pd.DataFrame({
            "Date": idx,
            "1.0_5_1.5x_EMA2040_91_61": [val*2 for val in [100, 200, 300]]
        })
        df_opti.to_csv(self.opti_curves_path, index=False)

    def tearDown(self):
        """
        Restore original working directory and remove the temporary directory.
        """
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_run_pp_analysis_integration(self):
        """
        Calls run_pp_analysis(...) with one plan and top_n=2,
        verifying that pp_curves.csv and a correlation CSV are created.
        """
        # We'll run from 2025-01-01..2025-01-03
        # concurrency='sync' to avoid pickling issues in unittests
        run_pp_analysis(
            tradeplans=[self.plan_str],
            dataset_path=self.dataset_path,
            start_date="2025-01-01",
            end_date="2025-01-03",
            opti_csv_path=self.opti_curves_path,
            init_capital=100000.0,
            top_n=2,
            debug=True,
            concurrency="sync"
        )

        # Check that pp_curves.csv is created
        self.assertTrue(
            os.path.exists("pp_curves.csv"),
            "Expected 'pp_curves.csv' to be written in the current directory."
        )
        df_pp_curves = pd.read_csv("pp_curves.csv")
        # Basic shape check => 3 rows => columns => ["Date", "1.0_5_1.5x_EMA2040"]
        self.assertEqual(df_pp_curves.shape[0], 3)
        self.assertIn(self.plan_str, df_pp_curves.columns)

        # Check correlation file => "pp_correlation_YYYYMMDD_HHMMSS.csv"
        corr_files = glob.glob("pp_correlation_*.csv")
        self.assertTrue(
            len(corr_files) > 0,
            "Expected at least one correlation CSV to be generated."
        )
        corr_files.sort(key=os.path.getmtime)
        newest_corr = corr_files[-1]
        df_corr = pd.read_csv(newest_corr)
        # Expect 1 row => for "1.0_5_1.5x_EMA2040_91_61"
        self.assertEqual(len(df_corr), 1)
        # Some basic column checks
        expected_cols = [
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ]
        self.assertListEqual(df_corr.columns.tolist(), expected_cols)

        print("Test run_pp_analysis_integration: PASSED.")
