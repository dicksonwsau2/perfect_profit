"""
test_compute_daily_pp.py

Unit tests for the updated compute_daily_pp function in pp_worker.py.
Now each day must have >= top_n trades or we raise ValueError.
"""

import unittest
import pandas as pd
from datetime import datetime
from pp_worker import compute_daily_pp, setup_logger


class TestComputeDailyPP(unittest.TestCase):
    """
    Tests compute_daily_pp to confirm:
      - If any day has < top_n trades, raise ValueError.
      - If after date range filtering the DataFrame is empty, raise ValueError.
      - Normal scenario: each day has >= top_n trades => sums top_n.
    """

    def setUp(self):
        """
        Optionally enable debug logs for the tests.
        """
        self.logger = setup_logger(debug=True)
        self.logger.debug("Starting TestComputeDailyPPWithMinTrades...")

    def test_day_with_fewer_trades_raises(self):
        """
        If one day has fewer than top_n trades, we raise ValueError immediately.
        Example: day1 has 5 trades, day2 has only 2 trades => top_n=3 => error.
        """
        data = {
            'EntryTime': [
                '2025-01-01 09:00:00',  # day1 => 3 trades
                '2025-01-01 09:30:00',
                '2025-01-01 10:00:00',
                '2025-01-02 09:00:00',  # day2 => 2 trades
                '2025-01-02 09:30:00',
            ],
            'ProfitLossAfterSlippage': [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        df = pd.DataFrame(data)
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        date_range = pd.date_range("2025-01-01", "2025-01-02")

        # day1 => 3 trades, day2 => 2 trades, top_n=3 => day2 insufficient
        with self.assertRaises(ValueError) as ctx:
            compute_daily_pp(df, date_range, top_n=3, debug=True)

        self.assertIn("only 2 trades, need >= 3", str(ctx.exception))

    def test_empty_after_filter_raises(self):
        """
        If everything is out-of-range => filtered df empty => ValueError
        """
        data = {
            'EntryTime': ['2025-01-05 09:00:00'],
            'ProfitLossAfterSlippage': [10.0]
        }
        df = pd.DataFrame(data)
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        date_range = pd.date_range("2025-01-01", "2025-01-02")  # No overlap

        with self.assertRaises(ValueError) as ctx:
            compute_daily_pp(df, date_range, top_n=1, debug=True)
        self.assertIn("No trades remain after date filter", str(ctx.exception))

    def test_top_n_okay_for_all_days(self):
        """
        If each day has >= top_n trades, we do normal summation.
        Example: day1 => 3 trades, day2 => 3 trades, top_n=2 => sums top2 each day
        """
        data = {
            'EntryTime': [
                '2025-01-01 09:00:00', '2025-01-01 09:30:00', '2025-01-01 10:00:00',
                '2025-01-02 09:00:00', '2025-01-02 09:30:00', '2025-01-02 10:00:00'
            ],
            # day1 PnL => [5, 2, 1] => top2 => 5+2=7
            # day2 PnL => [4, 3, -1] => top2 => 4+3=7
            'ProfitLossAfterSlippage': [5, 2, 1, 4, 3, -1]
        }
        df = pd.DataFrame(data)
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        date_range = pd.date_range("2025-01-01", "2025-01-02")

        result_series = compute_daily_pp(df, date_range, top_n=2, debug=True)

        # Expect 2 days in result => day1=7, day2=7
        self.assertEqual(len(result_series), 2, "We have trades on 2 distinct days.")
        day1 = pd.to_datetime("2025-01-01").date()
        day2 = pd.to_datetime("2025-01-02").date()

        self.assertAlmostEqual(result_series.loc[day1], 7.0)
        self.assertAlmostEqual(result_series.loc[day2], 7.0)

    def test_day_has_exactly_top_n(self):
        """
        If a day has exactly top_n trades, it is valid. We sum all of them.
        """
        data = {
            'EntryTime': [
                '2025-01-02 09:00:00',
                '2025-01-02 09:15:00',
                '2025-01-02 09:30:00',
            ],
            'ProfitLossAfterSlippage': [3.0, 2.0, 1.0]
        }
        df = pd.DataFrame(data)
        df['EntryTime'] = pd.to_datetime(df['EntryTime'])
        date_range = pd.date_range("2025-01-02", "2025-01-02")  # Single day

        # day => exactly 3 trades, top_n=3 => sum all => 6
        result_series = compute_daily_pp(df, date_range, top_n=3, debug=True)

        self.assertEqual(len(result_series), 1, "Only 1 day in range.")
        self.assertAlmostEqual(result_series.iloc[0], 6.0, places=4)


if __name__ == "__main__":
    unittest.main()
