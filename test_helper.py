"""
test_helper.py

Unit tests for the helper.py module, focusing on discover_plans_from_opti_csv().
"""

import unittest
import os
import tempfile
import pandas as pd

# Assuming your helper.py is in the same directory level; adjust import as needed:
from helper import discover_plans_from_opti_csv

class TestDiscoverPlansFromOptiCsv(unittest.TestCase):
    def setUp(self):
        """
        Creates a temporary directory for ephemeral CSV creation.
        We'll store an old working directory to restore it after the tests.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        self.opti_csv_path = os.path.join(self.temp_dir.name, "opti_curves.csv")

    def tearDown(self):
        """
        Cleanup: restore old working directory, remove temp directory.
        """
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_file_not_found(self):
        """
        If the file doesn't exist, discover_plans_from_opti_csv should raise FileNotFoundError.
        """
        non_existent = "this_file_does_not_exist.csv"
        with self.assertRaises(FileNotFoundError):
            discover_plans_from_opti_csv(non_existent, debug=True)

    def test_empty_prefix_list(self):
        """
        If the CSV has columns but none are valid prefix columns, we get an empty list.
        e.g. columns => ["Date","Foo"] => 'Foo' has < 4 segments => skip
        """
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=3),
            "Foo": [10, 20, 30]  # This column has <4 segments => skip
        })
        df.to_csv(self.opti_csv_path, index=False)

        result = discover_plans_from_opti_csv(self.opti_csv_path, debug=True)
        self.assertEqual(len(result), 0, "Expected no discovered prefixes.")

    def test_mixed_valid_and_invalid_columns(self):
        """
        Some columns are valid (>=4 segments), some are not.
        We also have a 'Date' column which must be dropped.
        """
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=3),
            "3.0_5_1.5x_EMA520_91_61": [100, 200, 300],  # valid => prefix => 3.0_5_1.5x_EMA520
            "FooBar": [1, 2, 3],                        # invalid => skip
            "2.0_10_1.0x_EMA540_66_99": [10, 20, 30],    # valid => prefix => 2.0_10_1.0x_EMA540
            "Short": [9, 9, 9],                         # e.g. 'Short' => skip
        })
        df.to_csv(self.opti_csv_path, index=False)

        result = discover_plans_from_opti_csv(self.opti_csv_path, debug=True)
        # Expect => ["2.0_10_1.0x_EMA540", "3.0_5_1.5x_EMA520"] sorted alphabetically
        self.assertListEqual(
            result,
            ["2.0_10_1.0x_EMA540", "3.0_5_1.5x_EMA520"],
            f"Unexpected discovered prefixes => {result}"
        )

    def test_no_date_column_scenario(self):
        """
        If there's no 'Date' column, that's fine. We just parse all columns that
        have >=4 segments. Others are skipped.
        """
        df = pd.DataFrame({
            "4.0_5_1.5x_EMA520_111_222": [10, 10, 10],
            "X_Y_Z": [1, 2, 3],  # <4 segments => skip
        })
        df.to_csv(self.opti_csv_path, index=False)

        result = discover_plans_from_opti_csv(self.opti_csv_path, debug=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "4.0_5_1.5x_EMA520")

    def test_duplicate_prefixes(self):
        """
        If the CSV has multiple columns with the *same* prefix but different suffixes,
        e.g. "3.0_5_1.5x_EMA520_91_61" and "3.0_5_1.5x_EMA520_55_75",
        we only want the prefix once. So the result is unique.
        """
        df = pd.DataFrame({
            "Date": pd.date_range("2025-01-01", periods=3),
            "3.0_5_1.5x_EMA520_91_61": [1, 2, 3],
            "3.0_5_1.5x_EMA520_55_75": [4, 5, 6],   # same prefix, different suffix
        })
        df.to_csv(self.opti_csv_path, index=False)

        result = discover_plans_from_opti_csv(self.opti_csv_path, debug=True)
        # Should be only 1 distinct prefix => "3.0_5_1.5x_EMA520"
        self.assertListEqual(result, ["3.0_5_1.5x_EMA520"])
        # check no duplicates

if __name__ == "__main__":
    unittest.main()
