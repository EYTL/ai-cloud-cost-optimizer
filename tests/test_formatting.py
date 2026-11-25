from formatting import fmt_usd


def test_normal_amounts_use_two_decimals():
    assert fmt_usd(12) == "$12.00"
    assert fmt_usd(0.02) == "$0.02"
    assert fmt_usd(3461.61) == "$3,461.61"


def test_tiny_amounts_use_four_decimals_instead_of_rounding_to_zero():
    # Regression: the dashboard used to show $0.0000 for NASA costs.
    assert fmt_usd(0.0071) == "$0.0071"
    assert fmt_usd(0.0004) == "$0.0004"


def test_zero_is_two_decimals():
    assert fmt_usd(0) == "$0.00"


def test_negative_values():
    assert fmt_usd(-0.5) == "-$0.50"
    assert fmt_usd(-0.0071) == "-$0.0071"


def test_signed_option_adds_plus_for_positive_only():
    assert fmt_usd(1.5, signed=True) == "+$1.50"
    assert fmt_usd(-1.5, signed=True) == "-$1.50"
