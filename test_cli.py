"""
test_cli.py

Unit tests for the command-line interface in perfect_profit.py,
using Approach A (a main_cli() function). We patch sys.argv and
mock run_pp_analysis to confirm the CLI passes correct arguments.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
import perfect_profit  # This imports the module but does not run main_cli automatically.


class TestPerfectProfitCLI(unittest.TestCase):
    @patch("perfect_profit.run_pp_analysis")
    def test_cli_with_args(self, mock_run_pp):
        """
        Patches sys.argv and calls perfect_profit.main_cli() to simulate
        a CLI invocation. Then checks that run_pp_analysis was called.
        """
        test_args = [
            "perfect_profit.py",
            "--dataset_path", "/some/dataset",
            "--start_date", "2025-01-01",
            "--end_date", "2025-01-05",
            "--opti_csv_path", "/some/opti.csv",
            "--init_capital", "123456",
            "--top_n", "5",
            "--debug",
            "--concurrency", "thread",
            "--tradeplan", "1.0_5_1.5x_EMA2040",
            "--tradeplan", "2.0_10_2.0x_EMA540"
        ]

        with patch.object(sys, "argv", test_args):
            perfect_profit.main_cli()

        # Ensure run_pp_analysis was indeed called once
        mock_run_pp.assert_called_once()

        # Grab the call args
        ((call_kwargs), _) = mock_run_pp.call_args
        # or for more direct: mock_run_pp.call_args[1] for keyword args
        # but here presumably arguments are positional => let's check carefully

        # We can do mock_run_pp.call_args.kwargs in python 3.8+ or do
        call_args, call_kwargs = mock_run_pp.call_args

        # Check each argument passed into run_pp_analysis
        self.assertIn("tradeplans", call_kwargs)
        self.assertListEqual(call_kwargs["tradeplans"], ["1.0_5_1.5x_EMA2040", "2.0_10_2.0x_EMA540"])
        self.assertEqual(call_kwargs["dataset_path"], "/some/dataset")
        self.assertEqual(call_kwargs["start_date"], "2025-01-01")
        self.assertEqual(call_kwargs["end_date"], "2025-01-05")
        self.assertEqual(call_kwargs["opti_csv_path"], "/some/opti.csv")
        self.assertEqual(call_kwargs["init_capital"], 123456.0)
        self.assertEqual(call_kwargs["top_n"], 5)
        self.assertTrue(call_kwargs["debug"])
        self.assertEqual(call_kwargs["concurrency"], "thread")


if __name__ == "__main__":
    unittest.main()
