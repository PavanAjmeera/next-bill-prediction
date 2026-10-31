import numpy as np
import pandas as pd

from src.data.clean import clean_bills, clean_customers
from src.data.generate import energy_charge
from src.features.build import FEATURES, build_model_table
from src.model.evaluate import error_metrics


def test_slab_tariff():
    month = pd.Period("2025-01", freq="M")
    # 150 units: 100 x 3.00 + 50 x 4.50 = 525
    assert energy_charge(150, "Domestic", month) == 525
    # after the April 2025 hike, everything is 5% higher
    assert round(energy_charge(150, "Domestic", pd.Period("2025-04", freq="M")), 2) == 551.25


def test_cleaning_fixes_duplicates_negatives_and_regions():
    customers = pd.DataFrame({"customer_id": ["C1"], "region": [" URBAN"]})
    clean_c, _ = clean_customers(customers)
    assert clean_c.loc[0, "region"] == "Urban"

    bills = pd.DataFrame({
        "customer_id": ["C1"] * 4,
        "bill_month": ["2024-01", "2024-02", "2024-02", "2024-03"],   # Feb duplicated
        "units_kwh": [100, -1, -1, 120],                               # -1 is impossible
        "bill_amount": [400.0, 450.0, 450.0, 500.0],
    })
    clean_b, report = clean_bills(bills)
    assert len(clean_b) == 3
    assert report["duplicate_bills_removed"] == 1
    assert clean_b["units_kwh"].tolist() == [100, 110, 120]           # filled from neighbours


def make_history(n_months=16):
    months = pd.period_range("2024-01", periods=n_months, freq="M").astype(str)
    bills = pd.DataFrame({
        "customer_id": "C1", "bill_month": months,
        "units_kwh": np.arange(n_months) * 10 + 100.0,
        "bill_amount": np.arange(n_months) * 100 + 1000.0,       # bill = 1000, 1100, 1200, ...
        "avg_temp_c": 25.0, "reading_type": "Actual", "days_to_pay": 10,
    })
    customers = pd.DataFrame({
        "customer_id": ["C1"], "connection_type": ["Domestic"], "region": ["Urban"],
        "sanctioned_load_kw": [2], "household_size": [3], "has_ac": [1],
        "meter_type": ["Smart"], "connection_start": ["2020-01"],
    })
    return bills, customers


def test_features_use_only_the_past():
    bills, customers = make_history()
    table = build_model_table(bills, customers).set_index("predict_month")
    row = table.loc["2025-04"]            # standing at the end of March 2025 (bill = 2,400)
    assert row["last_bill"] == 2400       # March's bill, not April's
    assert row["next_bill"] == 2500       # April's bill is only the answer
    assert row["bill_avg_3m"] == 2300     # Jan, Feb, Mar 2025
    assert row["bill_same_month_last_year"] == 1300   # April 2024
    assert "next_bill" not in FEATURES


def test_customers_need_three_months_of_history():
    bills, customers = make_history()
    table = build_model_table(bills, customers)
    assert (table["history_months"] >= 3).all()


def test_wape():
    actual = pd.Series([1000.0, 2000.0, 5000.0])
    predicted = pd.Series([900.0, 2200.0, 4700.0])
    assert error_metrics(actual, predicted)["WAPE"] == 0.075
