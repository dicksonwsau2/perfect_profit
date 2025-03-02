import unittest
import pandas as pd
from unittest.mock import patch, mock_open
import yaml
from pp_worker import create_equity_curve

class TestCreateEquityCurve(unittest.TestCase):
    """
    Tests the `create_equity_curve` function to ensure it calculates the equity curve correctly.
    """

    def setUp(self):
        self.init_capital = 100000.0
        self.plan_params = {
            'premium': '1.0',
            'width': '20',
            'stop_loss': '1.5x',
            'ema': 'EMA2040'
        }
        self.daily_pp = pd.Series([1.5, -0.5, 2.0, 1.0], index=pd.date_range('2025-01-01', periods=4))

    def test_create_equity_curve_without_bp_adjustment(self):
        """
        Test the creation of equity curve without BP adjustment (default behavior).
        """
        equity_curve = create_equity_curve(self.daily_pp, self.init_capital, self.plan_params, bp_adjusted=False, debug=True)
        self.assertEqual(equity_curve.attrs['plan_params'], self.plan_params)

        expected_values = self.init_capital + self.daily_pp.cumsum()
        pd.testing.assert_series_equal(equity_curve, expected_values, check_dtype=False)

    @patch('pp_worker.setup_logger')
    @patch('pp_worker.load_config')  # <--- Patch here, not in perfect_profit
    def test_create_equity_curve_with_bp_adjustment(self, mock_load_config, mock_logger):
        """
        Test the creation of equity curve with BP adjustment applied.
        """
        mock_config = {'tradeplan': {'MAX_WIDTH': 55}}
        mock_load_config.return_value = mock_config

        equity_curve = create_equity_curve(
            self.daily_pp, self.init_capital, self.plan_params, bp_adjusted=True, debug=True
        )

        scaling_factor = mock_config['tradeplan']['MAX_WIDTH'] / float(self.plan_params['width'])
        scaled_daily_pp = self.daily_pp * scaling_factor
        expected_values = self.init_capital + scaled_daily_pp.cumsum()

        pd.testing.assert_series_equal(equity_curve, expected_values, check_dtype=False)

if __name__ == "__main__":
    unittest.main()
