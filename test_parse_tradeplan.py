"""
test_parse_tradeplan.py

Unit tests for parse_tradeplan from pp_worker.py. 
Here, we do NOT rely on parse_tradeplan to check valid premium/width/ema, 
but we do so in the tests themselves after parse.
"""

import unittest
from pp_worker import parse_tradeplan


class TestParseTradeplan(unittest.TestCase):
    """
    Tests parse_tradeplan with no built-in validations,
    but verifying the output meets our domain rules:
      - premium in [1.0..5.0]
      - width in {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 100, 20-25-30, 45-50-55}
      - 4 segments must exist
    Note: no check for EMA in the function or tests.
    """

    # Valid sets / ranges
    VALID_WIDTHS = {
        "5", "10", "15", "20", "25", "30", "35",
        "40", "45", "50", "55", "100", 
        "20-25-30", "45-50-55"
    }

    def test_valid_parse_within_range(self):
        """
        Example of a valid parse: 
        premium=2.5, width=25, stop_loss=1.5x, ema=someema
        Then we check premium in [1..5], width in valid set, etc.
        """
        tradeplan_str = "2.5_25_1.5x_EMAwhatever"
        result = parse_tradeplan(tradeplan_str, debug=False)

        # Basic segments check
        self.assertEqual(result['premium'], "2.5")
        self.assertEqual(result['width'], "25")
        self.assertEqual(result['stop_loss'], "1.5x")
        self.assertEqual(result['ema'], "EMAwhatever")

        # domain rules: premium in [1..5], width in our set
        premium_float = float(result['premium'])
        self.assertTrue(
            1.0 <= premium_float <= 5.0, 
            f"Premium {premium_float} must be in [1.0..5.0]."
        )
        self.assertIn(
            result['width'], 
            self.VALID_WIDTHS,
            f"Width {result['width']} not in {self.VALID_WIDTHS}"
        )

    def test_valid_parse_range_and_multiseg_width(self):
        """
        Example with multi-seg width: "20-25-30"
        """
        tradeplan_str = "3.0_20-25-30_2x_ArbitraryEMA"
        result = parse_tradeplan(tradeplan_str)

        self.assertEqual(result['premium'], "3.0")
        self.assertEqual(result['width'], "20-25-30")
        self.assertEqual(result['stop_loss'], "2x")
        self.assertEqual(result['ema'], "ArbitraryEMA")

        # domain checks
        premium_float = float(result['premium'])
        self.assertTrue(1.0 <= premium_float <= 5.0)
        self.assertIn(result['width'], self.VALID_WIDTHS)

    def test_invalid_premium_outside_range(self):
        """
        premium=5.5 => outside [1..5], so domain check fails, 
        even though parse_tradeplan is successful in splitting. 
        We'll fail the test to indicate domain rule violation.
        """
        tradeplan_str = "5.5_10_1.5x_something"
        result = parse_tradeplan(tradeplan_str)  # parse succeeds
        premium_float = float(result['premium'])
        self.assertFalse(
            1.0 <= premium_float <= 5.0,
            f"Premium {premium_float} is out of [1.0..5.0]."
        )

    def test_invalid_width_not_in_set(self):
        """
        e.g. width=45-50 => not in the set
        """
        tradeplan_str = "2.0_45-50_1.0x_EMA"
        result = parse_tradeplan(tradeplan_str)
        # parse succeeded, now domain check fails
        self.assertNotIn(
            result['width'], 
            self.VALID_WIDTHS,
            f"Width {result['width']} not in {self.VALID_WIDTHS}"
        )

    def test_few_segments(self):
        """
        parse_tradeplan raises ValueError if fewer than 4 segments exist.
        """
        tradeplan_str = "1.0_5_ema"
        with self.assertRaises(ValueError):
            parse_tradeplan(tradeplan_str)

    def test_empty_string(self):
        """
        parse_tradeplan raises ValueError if string is empty
        """
        tradeplan_str = ""
        with self.assertRaises(ValueError):
            parse_tradeplan(tradeplan_str)


if __name__ == "__main__":
    unittest.main()
