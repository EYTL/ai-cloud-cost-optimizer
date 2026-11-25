import io
from pathlib import Path

import pandas as pd
import pytest

from anomaly_model import detect_anomalies
from data_loader import SIMULATED_PRICE_PER_REQUEST, load_data, load_nasa_data, load_uploaded_file
from formatting import fmt_usd

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def nasa(monkeypatch):
    # load_nasa_data() reads its CSV from the working directory.
    monkeypatch.chdir(REPO_ROOT)
    return load_nasa_data()


def test_nasa_cost_is_requests_times_simulated_price(nasa):
    expected = nasa["count"] * SIMULATED_PRICE_PER_REQUEST
    assert (nasa["cost"] - expected).abs().max() < 1e-12


def test_nasa_cost_does_not_display_as_zero(nasa):
    # Regression: costs used to be ~$0.00002 and rendered as $0.0000.
    assert fmt_usd(nasa["cost"].mean()) not in ("$0.00", "$0.0000")
    assert nasa["cost"].sum() > 100


def test_nasa_has_expected_columns_and_load_classes(nasa):
    assert {"date", "count", "cost", "department"} <= set(nasa.columns)
    assert set(nasa["department"]) <= {"High Load", "Normal Load", "Low Load"}
    assert len(nasa) > 1000


def test_load_data_rejects_unknown_source():
    with pytest.raises(ValueError):
        load_data("not-a-source")


def test_upload_with_cost_column():
    csv = io.StringIO("date,cost,service\n2026-01-02,5.5,EC2\n2026-01-01,2.0,S3\n")
    df = load_uploaded_file(csv)
    assert list(df["cost"]) == [2.0, 5.5]            # sorted by date
    assert list(df["department"]) == ["S3", "EC2"]   # taken from the service column


def test_upload_without_cost_derives_it_from_request_counts():
    csv = io.StringIO("timestamp,requests\n2026-01-01 00:00,10\n2026-01-01 00:01,200\n")
    df = load_uploaded_file(csv)
    assert df["cost"].iloc[1] == pytest.approx(200 * SIMULATED_PRICE_PER_REQUEST)
    assert list(df["department"]) == ["Low Load", "High Load"]


def test_upload_without_date_column_is_rejected():
    with pytest.raises(ValueError, match="No date column"):
        load_uploaded_file(io.StringIO("cost,service\n1,EC2\n"))


def test_anomaly_detection_flags_an_obvious_spike():
    df = pd.DataFrame({"cost": [10.0] * 99 + [10_000.0]})
    result = detect_anomalies(df)
    assert result["anomaly"].iloc[-1] == -1
