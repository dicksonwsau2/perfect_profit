import unittest
import pandas as pd
from unittest.mock import patch
from pp_worker import create_equity_curve


class TestCreateEquityCurve(unittest.TestCase):
    """
    Tests the `create_equity_curve` function to ensure it calculates 
    the equity curve correctly, including new multi-width logic.
    """

    def setUp(self):
        self.init_capital = 100000.0
        self.plan_params = {
            'premium': '1.0',
            'width': '20',          # We'll override this as needed
            'stop_loss': '1.5x',
            'ema': 'EMA2040'
        }
        # A small daily PP series for illustration
        # index: 4 consecutive days
        self.daily_pp = pd.Series(
            [1.5, -0.5, 2.0, 1.0],
            index=pd.date_range('2025-01-01', periods=4)
        )

    def test_create_equity_curve_without_bp_adjustment(self):
        """
        Test the creation of equity curve without BP adjustment (default).
        """
        equity_curve = create_equity_curve(
            self.daily_pp,
            self.init_capital,
            self.plan_params,
            bp_adjusted=False,
            debug=True
        )
        self.assertEqual(equity_curve.attrs['plan_params'], self.plan_params)

        expected_values = self.init_capital + self.daily_pp.cumsum()
        pd.testing.assert_series_equal(equity_curve, expected_values, check_dtype=False)

    @patch('pp_worker.setup_logger')
    @patch('pp_worker.load_config')
    def test_create_equity_curve_with_bp_adjustment_numeric_width(
        self, mock_load_config, mock_logger
    ):
        """
        Test with a numeric width (e.g. '20'), ensuring the scaling factor and *100 are applied.
        """
        mock_load_config.return_value = {'tradeplan': {'MAX_WIDTH': 55}}

        # width='20' => numeric_width=20 => scaling_factor=55/20=2.75
        # Then daily_pp * 2.75 * 100 => daily_pp * 275
        # daily_pp: [1.5, -0.5, 2.0, 1.0]
        # scaled:  [412.5, -137.5, 550.0, 275.0]
        # cumsum:  [412.5, 275.0, 825.0, 1100.0]
        # final => init_capital + cumsum => [100412.5, 100275.0, 100825.0, 101100.0]
        equity_curve = create_equity_curve(
            self.daily_pp,
            self.init_capital,
            self.plan_params,  # width='20'
            bp_adjusted=True,
            debug=True
        )

        # Build the same expected by manual calc
        scaling_factor = 55 / float(self.plan_params['width'])
        expected_daily = self.daily_pp * scaling_factor * 100
        expected_equity = self.init_capital + expected_daily.cumsum()

        pd.testing.assert_series_equal(equity_curve, expected_equity, check_names=False)
        self.assertEqual(equity_curve.attrs['plan_params'], self.plan_params)

    @patch('pp_worker.setup_logger')
    @patch('pp_worker.load_config')
    def test_create_equity_curve_multiwidth_20_25_30(
        self, mock_load_config, mock_logger
    ):
        """
        Test with '20-25-30' (treated as numeric_width=30).
        """
        mock_load_config.return_value = {'tradeplan': {'MAX_WIDTH': 55}}
        plan_params = self.plan_params.copy()
        plan_params['width'] = '20-25-30'  # Multiwidth

        # numeric_width=30 => scaling_factor=55/30=1.8333...
        # daily_pp => [1.5, -0.5, 2.0, 1.0]
        # multiply => [1.5*1.8333..*100, -0.5*..., 2.0*..., 1.0*... ]
        # i.e. ~ [275, -91.666..., 366.666..., 183.333...]
        # cumsum => [275, 183.333..., 550, 733.333...]
        # final => add init_capital => [100275, 100183.333..., 100550, 100733.333...]
        equity_curve = create_equity_curve(
            self.daily_pp,
            self.init_capital,
            plan_params,
            bp_adjusted=True,
            debug=True
        )

        numeric_width = 30.0
        scaling_factor = 55 / numeric_width
        expected_daily = self.daily_pp * scaling_factor * 100
        expected_equity = self.init_capital + expected_daily.cumsum()

        pd.testing.assert_series_equal(equity_curve, expected_equity, check_names=False)
        self.assertEqual(equity_curve.attrs['plan_params'], plan_params)

    @patch('pp_worker.setup_logger')
    @patch('pp_worker.load_config')
    def test_create_equity_curve_multiwidth_45_50_55(
        self, mock_load_config, mock_logger
    ):
        """
        Test with '45-50-55' (treated as numeric_width=55).
        """
        mock_load_config.return_value = {'tradeplan': {'MAX_WIDTH': 55}}
        plan_params = self.plan_params.copy()
        plan_params['width'] = '45-50-55'  # Multiwidth

        # numeric_width=55 => scaling_factor=55/55=1 => daily_pp * 1 * 100
        # => daily_pp * 100 => [150, -50, 200, 100]
        # cumsum => [150, 100, 300, 400]
        # final => add init_capital => [100150, 100100, 100300, 100400]
        equity_curve = create_equity_curve(
            self.daily_pp,
            self.init_capital,
            plan_params,
            bp_adjusted=True,
            debug=True
        )

        numeric_width = 55.0
        scaling_factor = 55 / numeric_width  # => 1
        expected_daily = self.daily_pp * scaling_factor * 100
        expected_equity = self.init_capital + expected_daily.cumsum()

        pd.testing.assert_series_equal(equity_curve, expected_equity, check_names=False)
        self.assertEqual(equity_curve.attrs['plan_params'], plan_params)

    @patch('pp_worker.setup_logger')
    @patch('pp_worker.load_config')
    def test_create_equity_curve_invalid_width_raises(self, mock_load_config, mock_logger):
        """
        Test that an unrecognized width string raises a ValueError.
        """
        mock_load_config.return_value = {'tradeplan': {'MAX_WIDTH': 55}}
        plan_params = self.plan_params.copy()
        plan_params['width'] = 'invalid-width-string'

        with self.assertRaises(ValueError) as cm:
            create_equity_curve(
                self.daily_pp,
                self.init_capital,
                plan_params,
                bp_adjusted=True,
                debug=True
            )
        self.assertIn("Unrecognized width format", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
