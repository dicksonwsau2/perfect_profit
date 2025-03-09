"""
test_run_pp_analysis.py

Integration tests for run_pp_analysis(...) in perfect_profit.py.
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
    Integration tests for run_pp_analysis.
    We'll create a minimal dataset CSV in dataset_path and a minimal
    optimized CSV, then let run_pp_analysis do its work, ensuring
    config.yaml is also present in the ephemeral directory.
    """

    def setUp(self):
        """
        Create a temporary directory as dataset_path and ephemeral environment.
        We'll also switch our working directory to it so that the final
        'pp_correlation_*.csv' ends up ephemeral as well.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        # 1) Create a minimal config.yaml
        config_yaml_path = os.path.join(self.temp_dir.name, "config.yaml")
        with open(config_yaml_path, "w") as f:
            f.write("tradeplan:\n  MAX_WIDTH: 55\n")

        # 2) Create a "datasets" folder for the CSV for the plan
        self.dataset_path = os.path.join(self.temp_dir.name, "datasets")
        os.makedirs(self.dataset_path, exist_ok=True)

    def tearDown(self):
        """
        Restore original working directory and remove the temporary directory.
        """
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_run_pp_analysis_integration(self):
        """
        Provides a single plan => "1.0_5_1.5x_EMA2040" in tradeplans,
        verifying that pp_curves.csv and a correlation CSV are created.
        """

        # (A) dataset CSV => "Dataset-Trades_1.0_5_1.5x_EMA2040_test.csv"
        plan_str = "1.0_5_1.5x_EMA2040"
        dataset_csv_name = f"Dataset-Trades_{plan_str}_test.csv"
        dataset_csv_path = os.path.join(self.dataset_path, dataset_csv_name)

        # Write minimal 3-day, 2-trades-per-day
        data_rows = []
        base_date = datetime(2025, 1, 1, 9, 0)
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
        df_dataset.to_csv(dataset_csv_path, index=False)

        # (B) optimized CSV => column => "1.0_5_1.5x_EMA2040_91_61"
        opti_curves_path = os.path.join(self.temp_dir.name, "opti_curves.csv")
        idx = pd.date_range("2025-01-01", periods=3, freq='D')
        df_opti = pd.DataFrame({
            "Date": idx,
            "1.0_5_1.5x_EMA2040_91_61": [val*2 for val in [100, 200, 300]]
        })
        df_opti.to_csv(opti_curves_path, index=False)

        # (C) call run_pp_analysis => pass the plan explicitly => no auto-discover
        run_pp_analysis(
            tradeplans=[plan_str],
            dataset_path=self.dataset_path,
            start_date="2025-01-01",
            end_date="2025-01-03",
            opti_csv_path=opti_curves_path,
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
        # shape => 3 rows => columns => ["Date", "1.0_5_1.5x_EMA2040"]
        self.assertEqual(df_pp_curves.shape[0], 3)
        self.assertIn(plan_str, df_pp_curves.columns)

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
        # columns
        expected_cols = [
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ]
        self.assertListEqual(df_corr.columns.tolist(), expected_cols)

    def test_run_pp_analysis_auto_discover(self):
        """
        Provides NO tradeplans => run_pp_analysis should auto-discover
        from 'opti_curves.csv' columns and produce pp_curves.csv, correlation CSV.
        """
        # We'll call plan => "2.0_5_1.5x_EMA520"
        plan_str = "2.0_5_1.5x_EMA520"
        dataset_csv_name = f"Dataset-Trades_{plan_str}_test.csv"
        dataset_csv_path = os.path.join(self.dataset_path, dataset_csv_name)

        # Minimal 3-day dataset for plan_str
        data_rows = []
        base_date = datetime(2025, 1, 1, 9, 0)
        for day_offset in range(3):
            for minute_offset in [0, 15]:
                entry_dt = base_date.replace(hour=9, minute=minute_offset) + \
                           pd.Timedelta(days=day_offset)
                profit = 5.0 + day_offset*1.5 + minute_offset*0.1
                data_rows.append({
                    "EntryTime": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "ProfitLossAfterSlippage": profit
                })
        df_dataset = pd.DataFrame(data_rows)
        df_dataset.to_csv(dataset_csv_path, index=False)

        # The opti CSV => column => "2.0_5_1.5x_EMA520_99_88" => e.g.
        opti_curves_path = os.path.join(self.temp_dir.name, "opti_curves.csv")
        idx = pd.date_range("2025-01-01", periods=3, freq='D')
        df_opti = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520_99_88": [200, 210, 220]
        })
        df_opti.to_csv(opti_curves_path, index=False)

        # Now we pass tradeplans=[] => auto-discover => "2.0_5_1.5x_EMA520"
        run_pp_analysis(
            tradeplans=[],  # empty => auto-discover from opti CSV
            dataset_path=self.dataset_path,
            start_date="2025-01-01",
            end_date="2025-01-03",
            opti_csv_path=opti_curves_path,
            init_capital=100000.0,
            top_n=2,
            debug=True,
            concurrency="sync"
        )

        # verify => "pp_curves.csv" => columns => ["Date","2.0_5_1.5x_EMA520"]
        self.assertTrue(os.path.exists("pp_curves.csv"), "No pp_curves.csv found!")
        df_pp = pd.read_csv("pp_curves.csv")
        self.assertIn(plan_str, df_pp.columns, "plan not discovered?!")

        # correlation => "pp_correlation_YYYYMMDD_*.csv"
        corr_files = glob.glob("pp_correlation_*.csv")
        self.assertTrue(len(corr_files)>0, "No correlation CSV found!")
        corr_files.sort(key=os.path.getmtime)
        newest_corr = corr_files[-1]
        df_corr = pd.read_csv(newest_corr)
        self.assertEqual(len(df_corr), 1)  # 1 plan
        # check columns
        expected_cols = [
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ]
        self.assertListEqual(df_corr.columns.tolist(), expected_cols)

        # verify the parted-out prefix => premium => "2.0", width => "5", ...
        row = df_corr.iloc[0]
        self.assertAlmostEqual(row["premium"], 2.0, places=1)
        self.assertAlmostEqual(row["width"], 5.0, places=1)
        self.assertEqual(row["SL"], "1.5x")
        self.assertEqual(row["EMA"], "EMA520")
        self.assertAlmostEqual(row["OPL"], 99.0, places=1)
        self.assertAlmostEqual(row["EPL"], 88.0, places=1)

if __name__ == "__main__":
    unittest.main()
