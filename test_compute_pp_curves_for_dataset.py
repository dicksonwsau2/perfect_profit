"""
test_compute_pp_curves_for_dataset.py

Unit tests for the compute_pp_curves_for_dataset function in perfect_profit.py.
Now includes an init_capital and top_n argument to ensure they're passed correctly.
"""

import unittest
import os
import tempfile
import pandas as pd
from unittest.mock import patch
from perfect_profit import compute_pp_curves_for_dataset


class TestComputePPCurvesForDataset(unittest.TestCase):
    """
    Tests compute_pp_curves_for_dataset to ensure it:
      - Calls worker_pp_computation for each tradeplan.
      - Writes the expected CSV output.
      - Passes init_capital and top_n to the worker correctly.
    """

    def setUp(self):
        """
        Create a temporary directory for test output and define sample inputs.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_csv_path = os.path.join(self.temp_dir.name, "pp_curves.csv")

        self.tradeplans = ["1.0_5_1.5x_EMA2040", "1.5_10_2.0x_EMA540"]
        self.dataset_path = "/path/to/dataset"  # Dummy path for testing
        self.start_date = "2025-01-01"
        self.end_date = "2025-01-05"
        self.sample_init_capital = 100000.0  # Example capital
        self.sample_top_n = 10               # Example top_n

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        self.temp_dir.cleanup()

    @patch('perfect_profit.worker_pp_computation')
    def test_compute_pp_curves_for_dataset(self, mock_worker):
        """
        Test that compute_pp_curves_for_dataset calls worker_pp_computation
        for each tradeplan, passes init_capital and top_n, and writes a CSV.
        We use concurrency='thread' to avoid pickling issues with MagicMock.
        """
        index = pd.date_range(start=self.start_date, end=self.end_date, freq='D')

        def side_effect_worker(
            plan,
            dataset_path,
            date_range,
            init_capital,
            top_n,
            debug=False
        ):
            # Ensure the dataset_path, init_capital, top_n match what we passed
            self.assertEqual(dataset_path, self.dataset_path)
            self.assertEqual(init_capital, self.sample_init_capital)
            self.assertEqual(top_n, self.sample_top_n)

            # Create a simple Series: [0..len(date_range)-1]
            data = range(len(date_range))
            series = pd.Series(data, index=date_range, name=plan)
            return series

        mock_worker.side_effect = side_effect_worker

        # Call the function under test with concurrency='thread'
        compute_pp_curves_for_dataset(
            tradeplans=self.tradeplans,
            dataset_path=self.dataset_path,
            start_date=self.start_date,
            end_date=self.end_date,
            init_capital=self.sample_init_capital,  # Pass our sample capital
            top_n=self.sample_top_n,                # Pass our sample top_n
            output_csv_path=self.output_csv_path,
            debug=True,
            concurrency='thread'
        )

        # Ensure the worker was called exactly once per tradeplan
        self.assertEqual(mock_worker.call_count, len(self.tradeplans))

        # Verify the output CSV now exists
        self.assertTrue(
            os.path.exists(self.output_csv_path),
            "The output CSV file was not created."
        )

        # Read the CSV and check the resulting DataFrame
        df_result = pd.read_csv(self.output_csv_path, index_col='date', parse_dates=True)
        self.assertListEqual(sorted(df_result.columns.tolist()), sorted(self.tradeplans))
        self.assertEqual(df_result.shape, (5, 2))

        # Verify data matches what our mock returned
        for plan in self.tradeplans:
            expected_series = pd.Series(range(5), index=index, name=plan)
            # Match the actual index name from CSV
            expected_series.index.name = "date"

            # Avoid freq mismatch check
            pd.testing.assert_series_equal(
                df_result[plan],
                expected_series,
                check_freq=False
            )


if __name__ == "__main__":
    unittest.main()
