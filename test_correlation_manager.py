"""
Integration and unit tests for the correlation_manager module.

This file includes:
  - Tests for load_and_filter_data, verify_date_coverage, parse_opti_column
  - Integration tests for compare_pp_and_optimized_curves using real
    temporary CSV files in ephemeral directories.
"""

import unittest
import os
import tempfile
import glob
import pandas as pd
from datetime import datetime
from correlation_manager import (
    load_and_filter_data,
    verify_date_coverage,
    parse_opti_column,
    compare_pp_and_optimized_curves
)


class TestLoadAndFilterData(unittest.TestCase):
    """
    Tests for the load_and_filter_data(...) function.
    Ensures that CSV loading with 'Date' as index_col and subsequent filtering
    by [start_date..end_date] is performed correctly.
    """

    def setUp(self):
        """
        Create a temporary directory and file, switching the current working
        directory to ensure any CSV writes/reads happen inside this ephemeral
        folder.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        self.csv_path = os.path.join(self.temp_dir.name, "test_pp.csv")

        # Create 5 days of data with "Date" as a column for load_and_filter_data
        idx = pd.date_range("2025-01-01", periods=5, freq='D')
        df = pd.DataFrame({
            "Date": idx,
            "colA": [10, 20, 30, 40, 50]
        })
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """
        Return to the original working directory and remove the temporary
        directory (and all files within).
        """
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_load_and_filter_data(self):
        """
        Filter from 2025-01-02..2025-01-04 => we expect 3 rows out of 5
        with colA => [20, 30, 40].
        """
        df_filtered = load_and_filter_data(
            csv_path=self.csv_path,
            start_date="2025-01-02",
            end_date="2025-01-04",
            debug=True
        )
        self.assertEqual(len(df_filtered), 3)
        self.assertListEqual(df_filtered["colA"].tolist(), [20, 30, 40])


class TestVerifyDateCoverage(unittest.TestCase):
    """
    Tests for verify_date_coverage(...). Ensures mismatching or matching
    sets of dates raise or pass as expected.
    """

    def test_same_coverage_ok(self):
        """
        Both DataFrames have identical date ranges => no error raised.
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')
        df1 = pd.DataFrame({"val1": [1, 2, 3, 4, 5]}, index=idx)
        df2 = pd.DataFrame({"val2": [6, 7, 8, 9, 10]}, index=idx)
        # Should not raise an error
        verify_date_coverage(df1, df2, debug=True)

    def test_mismatch_coverage(self):
        """
        df1 => 2025-01-01..01-05; df2 => 2025-01-02..01-06 => mismatch coverage,
        expecting ValueError about coverage differences.
        """
        idx1 = pd.date_range("2025-01-01", periods=5, freq='D')
        idx2 = pd.date_range("2025-01-02", periods=5, freq='D')
        df1 = pd.DataFrame({"val1": range(5)}, index=idx1)
        df2 = pd.DataFrame({"val2": range(5)}, index=idx2)

        with self.assertRaises(ValueError) as ctx:
            verify_date_coverage(df1, df2, debug=True)
        self.assertIn("Mismatched date coverage", str(ctx.exception))


class TestParseOptiColumn(unittest.TestCase):
    """
    Tests for parse_opti_column(...), verifying the 6-part logic for columns
    like '3.0_5_1.5x_EMA520_91_61'.
    """

    def test_valid_6_parts(self):
        """
        '3.14_5_1.5x_EMA520_91_61' => parse out:
          premium => '3.1' (float forced to 1 decimal),
          width => '5',
          SL => '1.5x',
          EMA => 'EMA520',
          OPL => '91',
          EPL => '61'
        """
        col_name = "3.14_5_1.5x_EMA520_91_61"
        premium, width, SL, EMA, OPL, EPL = parse_opti_column(col_name, debug=True)
        self.assertEqual(premium, "3.1")
        self.assertEqual(width, "5")
        self.assertEqual(SL, "1.5x")
        self.assertEqual(EMA, "EMA520")
        self.assertEqual(OPL, "91")
        self.assertEqual(EPL, "61")

    def test_invalid_num_parts(self):
        """
        Only 5 parts => must raise ValueError about needing EXACTLY 6 parts.
        """
        col_name = "2.0_5_1.5x_EMA520_99"
        with self.assertRaises(ValueError) as ctx:
            parse_opti_column(col_name, debug=True)
        self.assertIn("must have EXACTLY 6 parts", str(ctx.exception))

    def test_premium_parse_fail(self):
        """
        Premium is 'abc' => parse fails => 'Failed to parse 'abc' as float'
        """
        col_name = "abc_5_1.5x_EMA520_91_61"
        with self.assertRaises(ValueError) as ctx:
            parse_opti_column(col_name, debug=True)
        self.assertIn("Failed to parse 'abc' as float", str(ctx.exception))


class TestComparePPandOptimizedCurves(unittest.TestCase):
    """
    Integration tests for compare_pp_and_optimized_curves(...), verifying
    it can read real CSV files (with 'Date' as the first column), parse the
    plan prefix/suffix logic, and compute correlation. Uses ephemeral working
    directories for isolation.
    """

    def setUp(self):
        """
        Create a temporary directory and switch CWD so that any correlation
        CSV files (pp_correlation_*.csv) are written there and cleaned up
        afterwards.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)

        self.pp_csv_path = os.path.join(self.temp_dir.name, "pp_curves.csv")
        self.opti_csv_path = os.path.join(self.temp_dir.name, "opti_curves.csv")

    def tearDown(self):
        """
        Restore original CWD and remove the temporary directory, clearing
        all CSV files.
        """
        os.chdir(self.old_cwd)
        self.temp_dir.cleanup()

    def test_perfect_positive_corr(self):
        """
        5 days => Perfect +1 correlation: 
          PP => "2.0_5_1.5x_EMA520" => [10,20,30,40,50]
          Opti => "2.0_5_1.5x_EMA520_91_61" => [20,40,60,80,100] => x2
        Expects correlation ~ 1.0
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')
        df_pp = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520": [10, 20, 30, 40, 50]
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520_91_61": [val * 2 for val in [10,20,30,40,50]]
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        df_corr = compare_pp_and_optimized_curves(
            pp_csv_path=self.pp_csv_path,
            opti_csv_path=self.opti_csv_path,
            start_date="2025-01-01",
            end_date="2025-01-05",
            debug=True
        )

        self.assertEqual(len(df_corr), 1)
        row = df_corr.iloc[0]
        self.assertAlmostEqual(row["PearsonCorrelation"], 1.0, places=6)
        self.assertEqual(row["OptiPL"], 100.0)
        self.assertEqual(row["PPPL"], 50.0)

        corr_files = glob.glob("pp_correlation_*.csv")
        self.assertTrue(len(corr_files) > 0)

    def test_perfect_negative_corr(self):
        """
        5 days => Perfect -1 correlation:
          PP => "4.0_10_2.0x_EMA540" => [1,2,3,4,5]
          Opti => "4.0_10_2.0x_EMA540_91_61" => [5,4,3,2,1]
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')
        df_pp = pd.DataFrame({
            "Date": idx,
            "4.0_10_2.0x_EMA540": [1, 2, 3, 4, 5]
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx,
            "4.0_10_2.0x_EMA540_91_61": [5,4,3,2,1]
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        df_corr = compare_pp_and_optimized_curves(
            pp_csv_path=self.pp_csv_path,
            opti_csv_path=self.opti_csv_path,
            start_date="2025-01-01",
            end_date="2025-01-05",
            debug=True
        )
        self.assertEqual(len(df_corr), 1)
        row = df_corr.iloc[0]
        self.assertAlmostEqual(row["PearsonCorrelation"], -1.0, places=6)
        self.assertAlmostEqual(row["PPPL"], 5.0)
        self.assertAlmostEqual(row["OptiPL"], 1.0)

    def test_no_data_in_range(self):
        """
        5 days in CSV for 2025, but request 2026 => no overlap => 
        immediate "No data remain in PP CSV..." or "No data remain in OPTI CSV..."
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')
        df_pp = pd.DataFrame({
            "Date": idx,
            "3.0_5_2.0x_EMA540": [100, 200, 300, 400, 500]
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx,
            "3.0_5_2.0x_EMA540_91_61": [val + 50 for val in [100,200,300,400,500]]
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            compare_pp_and_optimized_curves(
                pp_csv_path=self.pp_csv_path,
                opti_csv_path=self.opti_csv_path,
                start_date="2026-01-01",
                end_date="2026-01-05",
                debug=True
            )
        err_str = str(ctx.exception)
        self.assertTrue(
            "No data remain in PP CSV after date filtering." in err_str
            or "No data remain in OPTI CSV after date filtering." in err_str
        )

    def test_partial_overlap_dates(self):
        """
        If PP has 5 days, but Opti has only 3 => coverage mismatch => ValueError.
        """
        idx_pp = pd.date_range("2025-01-01", periods=5, freq='D')
        idx_opti = pd.date_range("2025-01-02", periods=3, freq='D')
        df_pp = pd.DataFrame({
            "Date": idx_pp,
            "2.5_5_1.5x_EMA530": [11,22,33,44,55]
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx_opti,
            "2.5_5_1.5x_EMA530_91_61": [22,44,66]
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            compare_pp_and_optimized_curves(
                pp_csv_path=self.pp_csv_path,
                opti_csv_path=self.opti_csv_path,
                start_date="2025-01-01",
                end_date="2025-01-05",
                debug=True
            )
        self.assertIn("Mismatched date coverage", str(ctx.exception))

    def test_multiple_optiplans_with_matching_prefixes(self):
        """
        2 columns in PP, 2 columns in Opti => each prefix matches with a 
        different OPL/EPL suffix. We confirm parted-out columns are correct
        and correlation is +1.0 for each plan.
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')

        df_pp = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520":  [10, 20, 30, 40, 50],
            "3.0_5_1.5x_EMA520": [100,200,300,400,500],
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520_91_61":  [val*2 for val in [10,20,30,40,50]],
            "3.0_5_1.5x_EMA520_95_65":  [val*2 for val in [100,200,300,400,500]],
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        df_corr = compare_pp_and_optimized_curves(
            pp_csv_path=self.pp_csv_path,
            opti_csv_path=self.opti_csv_path,
            start_date="2025-01-01",
            end_date="2025-01-05",
            debug=True
        )

        self.assertEqual(len(df_corr), 2)
        expected_cols = [
            "opti_id","premium","width","SL","EMA","OPL","EPL",
            "PearsonCorrelation","PearsonPValue","OptiPL","PPPL"
        ]
        self.assertListEqual(df_corr.columns.tolist(), expected_cols)

        df_corr.sort_values("opti_id", inplace=True)
        row1 = df_corr.iloc[0]
        self.assertEqual(row1["premium"], "2.0")
        self.assertEqual(row1["width"], "5")
        self.assertEqual(row1["SL"], "1.5x")
        self.assertEqual(row1["EMA"], "EMA520")
        self.assertEqual(row1["OPL"], "91")
        self.assertEqual(row1["EPL"], "61")
        self.assertAlmostEqual(row1["PearsonCorrelation"], 1.0, places=6)
        self.assertEqual(row1["PPPL"], 50.0)
        self.assertEqual(row1["OptiPL"], 100.0)

        row2 = df_corr.iloc[1]
        self.assertEqual(row2["premium"], "3.0")
        self.assertEqual(row2["width"], "5")
        self.assertEqual(row2["SL"], "1.5x")
        self.assertEqual(row2["EMA"], "EMA520")
        self.assertEqual(row2["OPL"], "95")
        self.assertEqual(row2["EPL"], "65")
        self.assertAlmostEqual(row2["PearsonCorrelation"], 1.0, places=6)
        self.assertEqual(row2["PPPL"], 500.0)
        self.assertEqual(row2["OptiPL"], 1000.0)

        corr_files = glob.glob("pp_correlation_*.csv")
        self.assertTrue(len(corr_files) > 0)
        corr_files.sort(key=os.path.getmtime)
        newest_corr = corr_files[-1]
        df_written = pd.read_csv(newest_corr)
        self.assertListEqual(df_written.columns.tolist(), expected_cols)
        self.assertEqual(len(df_written), 2)

    def test_multiple_optiplans_prefix_not_found(self):
        """
        2 columns in PP, 2 in Opti => but one Opti prefix doesn't match PP => 
        expect a ValueError about 'not found in pp_df'.
        """
        idx = pd.date_range("2025-01-01", periods=5, freq='D')

        df_pp = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520": [10, 20, 30, 40, 50],
            "3.0_5_1.5x_EMA520": [100,200,300,400,500],
        })
        df_pp.to_csv(self.pp_csv_path, index=False)

        df_opti = pd.DataFrame({
            "Date": idx,
            "2.0_5_1.5x_EMA520_91_61": [val*2 for val in [10,20,30,40,50]],
            "4.0_5_1.5x_EMA520_91_61": [val*2 for val in [10,20,30,40,50]],
        })
        df_opti.to_csv(self.opti_csv_path, index=False)

        with self.assertRaises(ValueError) as ctx:
            compare_pp_and_optimized_curves(
                pp_csv_path=self.pp_csv_path,
                opti_csv_path=self.opti_csv_path,
                start_date="2025-01-01",
                end_date="2025-01-05",
                debug=True
            )
        err_msg = str(ctx.exception)
        self.assertIn("not found in pp_df", err_msg)
        self.assertIn("4.0_5_1.5x_EMA520_91_61", err_msg)


if __name__ == "__main__":
    unittest.main()
