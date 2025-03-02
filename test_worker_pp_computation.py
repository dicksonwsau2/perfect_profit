import unittest
from unittest.mock import patch
import pandas as pd
from pp_worker import worker_pp_computation

class TestWorkerPPComputation(unittest.TestCase):
    """
    Integration tests for the worker_pp_computation function,
    ensuring it handles:
      1) 10 days with no BP adjustment,
      2) 10 days with BP adjustment,
      3) A scenario where at least one day has < top_n trades => ValueError.
    """

    def setUp(self):
        # Common parameters
        self.tradeplan = "1.0_20_1.5x_EMA2040"  # e.g. premium=1.0, width=20, stop_loss=1.5x, ema=EMA2040
        self.dataset_path = "/dummy/path"
        self.init_capital = 100000.0
        self.top_n = 2
        # We'll set bp_adjusted in each test

        # We'll create a 20-row DataFrame => 2 trades per day for days 1..10
        # Each day i => date => 2025-01-i, 2 trades => PnL => [i+1, i+2].
        # That means day1 => PnL => [2,3], day2 => [3,4], ... day10 => [11,12].
        # day i => sum => (i+1)+(i+2)=2i+3 => we have top_n=2 => we sum them all
        # final cumsum -> see explanation in the test.
        rows = []
        for i in range(1, 11):  # day1..day10
            # Two trades
            rows.append({
                'EntryTime': f"2025-01-{i:02d} 09:00:00",
                'ProfitLossAfterSlippage': i+1
            })
            rows.append({
                'EntryTime': f"2025-01-{i:02d} 09:15:00",
                'ProfitLossAfterSlippage': i+2
            })
        self.mock_df_10days = pd.DataFrame(rows)

    @patch("pp_worker.read_dataset_for_plan")
    def test_no_bp_adjusted_10days(self, mock_read):
        """
        Tests worker_pp_computation with 10 days,
        each day has exactly 2 trades => top_n=2 => sums them all.
        No BP adjustment.
        """
        mock_read.return_value = self.mock_df_10days

        # We call worker_pp_computation with bp_adjusted=False
        eq_curve = worker_pp_computation(
            tradeplan=self.tradeplan,
            dataset_path=self.dataset_path,
            date_range=pd.date_range("2025-01-01", "2025-01-10"),
            init_capital=self.init_capital,
            top_n=self.top_n,
            bp_adjusted=False,  # No scaling
            debug=True
        )

        # Let's compute the expected cumsum:
        # For day i => sum => (i+1)+(i+2) => 2i+3
        # day1 => 2*1+3=5 => cumsum=5
        # day2 => 2*2+3=7 => cumsum=5+7=12
        # day3 => 2*3+3=9 => cumsum=12+9=21
        # day4 => 11 => cumsum=32
        # day5 => 13 => cumsum=45
        # day6 => 15 => cumsum=60
        # day7 => 17 => cumsum=77
        # day8 => 19 => cumsum=96
        # day9 => 21 => cumsum=117
        # day10=> 23 => cumsum=140
        # Then add init_capital => day1=>100005, day2=>100012,..., day10=>100140
        # Actually note carefully: day i => 2i+3 => day1=5, day2=7, day3=9, day4=11,
        # day5=13, day6=15, day7=17, day8=19, day9=21, day10=23
        # cumsum => day1=5, day2=12, day3=21, day4=32, day5=45, day6=60, day7=77, day8=96, day9=117, day10=140
        # plus init => day1=100005, day2=100012, day3=100021, day4=100032, day5=100045, day6=100060, day7=100077,
        # day8=100096, day9=100117, day10=100140
        expected_values = [
            100005.0,
            100012.0,
            100021.0,
            100032.0,
            100045.0,
            100060.0,
            100077.0,
            100096.0,
            100117.0,
            100140.0
        ]
        # Check final eq_curve
        for i, day in enumerate(range(1, 11), start=0):
            day_date = pd.to_datetime(f"2025-01-{day:02d}").date()
            self.assertAlmostEqual(eq_curve.loc[day_date], expected_values[i], places=4)

    @patch("pp_worker.load_config")
    @patch("pp_worker.read_dataset_for_plan")
    def test_bp_adjusted_10days(self, mock_read, mock_config):
        """
        Similar to above, but with bp_adjusted=True => scale daily PP by factor= MAX_WIDTH/width => e.g. 55/20=2.75
        """
        mock_read.return_value = self.mock_df_10days
        mock_config.return_value = {'tradeplan': {'MAX_WIDTH': 55}}

        eq_curve = worker_pp_computation(
            tradeplan=self.tradeplan,
            dataset_path=self.dataset_path,
            date_range=pd.date_range("2025-01-01", "2025-01-10"),
            init_capital=self.init_capital,
            top_n=self.top_n,
            bp_adjusted=True,  # scale
            debug=True
        )

        # factor=2.75 => day i => daily sum => 2i+3 => scaled => (2i+3)*2.75 => cumsum => plus init.
        # day1 =>5=>13.75 => cumsum=13.75 => day2 =>7=>19.25 => cumsum=33 => day3=>9=>24.75 => cumsum=57.75 ...
        # Let's do it carefully:
        # day i => raw sum=2i+3 => scaled => (2i+3)*2.75 => let's denote S_i => then cumsum => plus 100000
        # i=1 => raw=5 => scaled=13.75 => cumsum=13.75 => day2=7 =>19.25 => total=33 => day3=9=>24.75 => total=57.75
        # day4=11=>30.25 => total=88 => day5=13=>35.75 => total=123.75 => day6=15=>41.25 => total=165 => day7=17=>46.75 => total=211.75
        # day8=19=>52.25 => total=264 => day9=21=>57.75 => total=321.75 => day10=23=>63.25 => total=385
        # Then offset by init => day10 => 100385 => day1 => 100013.75 => etc.

        # We'll define an array expected_cumsum => [13.75, 33,57.75,88,123.75,165,211.75,264,321.75,385]
        # Then add 100000 => => day1=100013.75, day2=100033, day3=100057.75, day4=100088, day5=100123.75,
        # day6=100165, day7=100211.75, day8=100264, day9=100321.75, day10=100385
        expected_scaled_cum = [
            13.75, 33.0, 57.75, 88.0, 123.75, 165.0, 211.75, 264.0, 321.75, 385.0
        ]
        expected_final = [val + 100000.0 for val in expected_scaled_cum]

        for i, day in enumerate(range(1, 11)):
            day_date = pd.to_datetime(f"2025-01-{day:02d}").date()
            self.assertAlmostEqual(eq_curve.loc[day_date], expected_final[i], places=4)

    @patch("pp_worker.read_dataset_for_plan")
    def test_insufficient_trades_raises(self, mock_read):
        """
        We'll create a scenario where day5 has only 1 trade => top_n=2 => ValueError
        """
        # We'll copy the 10-day data but remove the second row for day5
        df_insufficient = self.mock_df_10days.copy()

        # day5 => i=5 => look for '2025-01-05' => remove one row
        # let's remove the last row for day5 => e.g. the row with ProfitLossAfterSlippage= i+2 => (5+2=7)
        # We'll do it by filtering out that exact row or so:
        cond = (df_insufficient['EntryTime'] == "2025-01-05 09:15:00")
        df_insufficient = df_insufficient[~cond].reset_index(drop=True)

        # Now day5 has only 1 trade => top_n=2 => should raise ValueError
        mock_read.return_value = df_insufficient

        with self.assertRaises(ValueError) as ctx:
            worker_pp_computation(
                tradeplan=self.tradeplan,
                dataset_path=self.dataset_path,
                date_range=pd.date_range("2025-01-01", "2025-01-10"),
                init_capital=self.init_capital,
                top_n=self.top_n,  # top_n=2 => day5 has only 1 => error
                bp_adjusted=False,
                debug=True
            )

        self.assertIn("only 1 trades, need >= 2", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
