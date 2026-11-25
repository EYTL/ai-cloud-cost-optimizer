"""Small display helpers shared by the dashboard (kept separate so they can be unit tested)."""


def fmt_usd(value, signed=False):
    """Format a dollar amount: 2 decimals normally, 4 for tiny real-world values."""
    prefix = "-" if value < 0 else ("+" if signed else "")
    amount = abs(float(value))
    digits = 4 if 0 < amount < 0.01 else 2
    return f"{prefix}${amount:,.{digits}f}"
