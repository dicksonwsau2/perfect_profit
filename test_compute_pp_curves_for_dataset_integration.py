"""
Integration tests for compute_pp_curves_for_dataset, using the real worker_pp_computation
and minimal CSV files that do NOT have a separate 'date' column.

We rely on 'EntryTime' for daily grouping, matching how the real code works.

We create:
  - Plan #1 ("1.0_10_1.5x_EMA2040") => 5 days, each day 2 trades => top_n=2 => success
  - Plan #2 ("2.0_5_2x_EMA520")     => 5 days, each day 2 trades => top_n=2 => success
  - Plan #3 ("3.0_5_2.5x_EMA540")  => day3 missing second trade => top_n=2 => triggers error

At the end, we check the final CSV thoroughly, verifying the daily sums, cumsums,
offset by init_capital.
"""

import unittest
import os
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from pp_compute import compute_pp_curves_for_dataset


class TestComputePPCurvesForDatasetIntegration(unittest.TestCase):
    """
    Integration tests verifying that compute_pp_curves_for_dataset runs the real
    worker computations, writes a CSV with columns = tradeplan names, and handles
    insufficient trades per day if top_n cannot be satisfied.
    """

    def setUp(self):
        """
        Create a temporary directory with 3 minimal CSV files:
          1) Plan #1: "1.0_10_1.5x_EMA2040"
          2) Plan #2: "2.0_5_2x_EMA520"
          3) Plan #3: "3.0_5_2.5x_EMA540" (missing second trade on day3 => triggers error)

        Each CSV has columns: [EntryTime, ProfitLossAfterSlippage].
        We'll run from 2025-01-01..2025-01-05 => top_n=2 => each day must have >=2 trades,
        except plan #3 fails on day3.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = self.temp_dir.name
        self.output_csv_path = os.path.join(self.dataset_path, "pp_curves.csv")

        self.start_date = "2025-01-01"
        self.end_date = "2025-01-05"
        self.init_capital = 100000.0
        self.top_n = 2

        self.plan1 = "1.0_10_1.5x_EMA2040"
        self.plan2 = "2.0_5_2x_EMA520"
        self.plan3 = "3.0_5_2.5x_EMA540"

        # We'll test success with plan1 & plan2, and an error scenario with plan3.
        self.tradeplans_good = [self.plan1, self.plan2]
        self.tradeplans_bad = [self.plan1, self.plan3]

        # Helper function to generate 'EntryTime' strings: day i => 2025-01-01 + i days
        def dt_str_for_day(i, hour, minute):
            base_dt = datetime(2025, 1, 1, hour, minute) + timedelta(days=i)
            return base_dt.strftime("%Y-%m-%d %H:%M:%S")

        # --- CSV for Plan #1 ---
        csv1_name = f"Dataset-Trades_{self.plan1}_foo.csv"
        csv1_path = os.path.join(self.dataset_path, csv1_name)
        rows1 = []
        # Each of 5 days => 2 trades => top_n=2 is satisfied
        for i in range(5):
            rows1.append({
                "EntryTime": dt_str_for_day(i, 9, 0),
                "ProfitLossAfterSlippage": 5.0 + i
            })
            rows1.append({
                "EntryTime": dt_str_for_day(i, 9, 15),
                "ProfitLossAfterSlippage": 10.0 + i
            })
        df1 = pd.DataFrame(rows1)
        df1.to_csv(csv1_path, index=False)

        # --- CSV for Plan #2 ---
        csv2_name = f"Dataset-Trades_{self.plan2}_bar.csv"
        csv2_path = os.path.join(self.dataset_path, csv2_name)
        rows2 = []
        for i in range(5):
            rows2.append({
                "EntryTime": dt_str_for_day(i, 10, 0),
                "ProfitLossAfterSlippage": 2.0 + i
            })
            rows2.append({
                "EntryTime": dt_str_for_day(i, 10, 15),
                "ProfitLossAfterSlippage": 4.0 + i
            })
        df2 = pd.DataFrame(rows2)
        df2.to_csv(csv2_path, index=False)

        # --- CSV for Plan #3 ---
        # On day3 => only 1 trade => triggers ValueError for top_n=2
        csv3_name = f"Dataset-Trades_{self.plan3}_baz.csv"
        csv3_path = os.path.join(self.dataset_path, csv3_name)
        rows3 = []
        for i in range(5):
            if i == 2:  # day3
                rows3.append({
                    "EntryTime": dt_str_for_day(i, 11, 0),
                    "ProfitLossAfterSlippage": 7.0 + i
                })
            else:
                rows3.append({
                    "EntryTime": dt_str_for_day(i, 11, 0),
                    "ProfitLossAfterSlippage": 7.0 + i
                })
                rows3.append({
                    "EntryTime": dt_str_for_day(i, 11, 15),
                    "ProfitLossAfterSlippage": 8.0 + i
                })
        df3 = pd.DataFrame(rows3)
        df3.to_csv(csv3_path, index=False)

    def tearDown(self):
        """
        Remove the temporary directory after each test.
        """
        self.temp_dir.cleanup()

    def test_integration_success(self):
        """
        Plan1 & Plan2 => each day i has exactly 2 trades => top_n=2 => success => final CSV is created.
        We'll do the final math & check each day's equity for both plans.

        Detailed math for plan1 (just conceptual):
          day i => (5.0+i) + (10.0+i) = 15 + 2i
          partial cumsum => [15, 32, 51, 72, 95]
          offset => add init_capital=100000 => ...
        Similarly for plan2 => day i => (2.0+i) + (4.0+i) = 6 + 2i => ...
        """
        compute_pp_curves_for_dataset(
            tradeplans=self.tradeplans_good,
            dataset_path=self.dataset_path,
            start_date=self.start_date,
            end_date=self.end_date,
            init_capital=self.init_capital,
            top_n=self.top_n,
            output_csv_path=self.output_csv_path,
            bp_adjusted=False,
            concurrency="process",
            debug=True
        )

        self.assertTrue(
            os.path.exists(self.output_csv_path),
            "No CSV created on success!"
        )
        df_res = pd.read_csv(self.output_csv_path, parse_dates=["Date"])
        self.assertListEqual(
            sorted(df_res.columns.tolist()),
            sorted(["Date", self.plan1, self.plan2])
        )
        self.assertEqual(len(df_res), 5)

        plan1_cumsum = [15, 32, 51, 72, 95]
        plan2_cumsum = [6, 14, 24, 36, 50]

        for i in range(5):
            day_dt = pd.to_datetime(self.start_date) + pd.Timedelta(days=i)
            row = df_res.loc[df_res["Date"] == day_dt]
            self.assertFalse(
                row.empty,
                f"No final row for day {day_dt.date()}"
            )

            val_plan1 = row[self.plan1].iloc[0]
            val_plan2 = row[self.plan2].iloc[0]

            expected1 = 100000.0 + plan1_cumsum[i]
            expected2 = 100000.0 + plan2_cumsum[i]
            self.assertAlmostEqual(
                val_plan1, expected1, places=4,
                msg=f"Day {i+1}, plan1 => got {val_plan1}, expect {expected1}"
            )
            self.assertAlmostEqual(
                val_plan2, expected2, places=4,
                msg=f"Day {i+1}, plan2 => got {val_plan2}, expect {expected2}"
            )

    def test_integration_insufficient_trades(self):
        """
        Plan1 & Plan3 => day3 in plan3 has only 1 trade => top_n=2 => triggers ValueError
        No CSV should be created.
        """
        with self.assertRaises(ValueError) as ctx:
            compute_pp_curves_for_dataset(
                tradeplans=self.tradeplans_bad,
                dataset_path=self.dataset_path,
                start_date=self.start_date,
                end_date=self.end_date,
                init_capital=self.init_capital,
                top_n=self.top_n,
                output_csv_path=self.output_csv_path,
                bp_adjusted=False,
                concurrency="process",
                debug=True
            )
        self.assertIn("only 1 trades, need >= 2", str(ctx.exception))

        self.assertFalse(
            os.path.exists(self.output_csv_path),
            "CSV should not be created if an error is raised."
        )


if __name__ == "__main__":
    unittest.main()
