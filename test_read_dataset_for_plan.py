"""
test_read_dataset_for_plan.py

Unit tests for the read_dataset_for_plan function in pp_worker.py.
Ensures it handles:
  - No matched files
  - Multiple matched files
  - Empty CSV
  - Successful read with mock data resembling the sample CSV
"""

import unittest
import os
import pandas as pd
from unittest.mock import patch
from pp_worker import read_dataset_for_plan, setup_logger


class TestReadDatasetForPlan(unittest.TestCase):
    """
    Tests read_dataset_for_plan to confirm it:
      - Raises FileNotFoundError if no matched files or multiple files.
      - Raises FileNotFoundError if CSV is empty.
      - Returns a DataFrame for a single valid file with mock data.
    """

    def setUp(self):
        """
        Define a sample dataset path and plan_params.
        Also compute the expected file pattern.
        """
        # Optionally enable debug logs:
        logger = setup_logger(debug=True)
        logger.debug("Starting TestReadDatasetForPlan...")

        self.dataset_path = "/dummy/dataset/path"
        self.plan_params = {
            'premium': "1.0",
            'width': "10",
            'stop_loss': "1.5x",
            'ema': "EMA2040"
        }

        # The pattern read_dataset_for_plan forms:
        # Dataset-Trades_{premium}_{width}_{stop_loss}_{ema}_*.csv
        self.expected_pattern = os.path.join(
            self.dataset_path,
            "Dataset-Trades_1.0_10_1.5x_EMA2040_*.csv"
        )

    @patch("pp_worker.glob.glob")
    def test_no_matched_files(self, mock_glob):
        """
        If glob.glob returns an empty list, read_dataset_for_plan
        raises FileNotFoundError.
        """
        mock_glob.return_value = []

        with self.assertRaises(FileNotFoundError) as ctx:
            read_dataset_for_plan(
                dataset_path=self.dataset_path,
                plan_params=self.plan_params,
                debug=True
            )

        self.assertIn("No matching dataset CSV", str(ctx.exception))
        mock_glob.assert_called_once_with(self.expected_pattern)

    @patch("pp_worker.glob.glob")
    def test_multiple_matched_files(self, mock_glob):
        """
        If glob.glob returns multiple files, read_dataset_for_plan
        raises FileNotFoundError.
        """
        mock_glob.return_value = [
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_123.csv",
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_456.csv"
        ]

        with self.assertRaises(FileNotFoundError) as ctx:
            read_dataset_for_plan(
                dataset_path=self.dataset_path,
                plan_params=self.plan_params,
                debug=True
            )

        self.assertIn("Multiple matching files", str(ctx.exception))
        mock_glob.assert_called_once_with(self.expected_pattern)

    @patch("pp_worker.glob.glob")
    @patch("pp_worker.pd.read_csv")
    def test_empty_csv_raises(self, mock_read_csv, mock_glob):
        """
        If read_csv returns an empty DataFrame, read_dataset_for_plan
        raises FileNotFoundError.
        """
        mock_glob.return_value = [
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_789.csv"
        ]
        # Mock an empty DataFrame
        mock_read_csv.return_value = pd.DataFrame()

        with self.assertRaises(FileNotFoundError) as ctx:
            read_dataset_for_plan(
                dataset_path=self.dataset_path,
                plan_params=self.plan_params,
                debug=True
            )

        self.assertIn("Data read is empty for the matched file", str(ctx.exception))

        mock_glob.assert_called_once_with(self.expected_pattern)
        mock_read_csv.assert_called_once_with(
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_789.csv",
            parse_dates=['date']
        )

    @patch("pp_worker.glob.glob")
    @patch("pp_worker.pd.read_csv")
    def test_successful_read(self, mock_read_csv, mock_glob):
        """
        If exactly one file matches and CSV is non-empty, it returns
        a DataFrame resembling the sample you provided.
        """
        # Exactly one matching file
        mock_glob.return_value = [
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_000.csv"
        ]

        # Create a mock DataFrame similar to your sample CSV
        test_data = {
            'date': [
                '2023-01-03 09:33:00',
                '2023-01-03 09:45:00',
                '2023-01-03 10:00:00'
            ],
            'TradeID': [13705628, 13705964, 13712284],
            'OptionType': ['P', 'P', 'C'],
            'Width': [10.0, 10.0, 10.0],
            'Premium': [1.0, 0.8, 0.9],
            'ProfitLoss': [-1.5, -1.2, 0.9],
            'ProfitLossAfterSlippage': [-1.5, -1.2, 0.9],
        }
        df_mock = pd.DataFrame(test_data)
        # Convert 'date' to datetime to mimic parse_dates
        df_mock['date'] = pd.to_datetime(df_mock['date'])

        mock_read_csv.return_value = df_mock

        df_result = read_dataset_for_plan(
            dataset_path=self.dataset_path,
            plan_params=self.plan_params,
            debug=True
        )

        # Ensure DataFrame is not empty
        self.assertFalse(df_result.empty, "Returned DataFrame should not be empty.")

        # Check shape => 3 rows, 7 columns
        self.assertEqual(df_result.shape, (3, 7))

        # Check columns
        expected_cols = [
            'date',
            'TradeID',
            'OptionType',
            'Width',
            'Premium',
            'ProfitLoss',
            'ProfitLossAfterSlippage'
        ]
        self.assertListEqual(df_result.columns.tolist(), expected_cols)

        # Confirm the call patterns
        mock_glob.assert_called_once_with(self.expected_pattern)
        mock_read_csv.assert_called_once_with(
            "/dummy/dataset/path/Dataset-Trades_1.0_10_1.5x_EMA2040_000.csv",
            parse_dates=['date']
        )


if __name__ == "__main__":
    unittest.main()
