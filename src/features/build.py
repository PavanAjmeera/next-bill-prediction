"""
Build the model table (layer 3).

One row = one customer standing at the END of month M, predicting the bill of month M+1.
Every feature uses only information known by the end of month M (no temporal leakage).

Run from the project root:  python -m src.features.build
"""
import numpy as np
import pandas as pd

from src.config import (
    FORECAST_MONTH,
    MIN_HISTORY_MONTHS,
    PROCESSED_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from src.data.generate import TARIFF_HIKE_FROM

LATE_AFTER_DAYS = 20

CATEGORICAL_FEATURES = ["connection_type", "region", "meter_type"]
NUMERIC_FEATURES = [
    # A. customer details
    "sanctioned_load_kw", "household_size", "has_ac", "customer_age_months",
    # B. recent history
    "last_bill", "last_units", "bill_avg_3m", "bill_avg_6m", "units_avg_3m",
    "units_trend", "bill_std_6m", "bill_same_month_last_year",
    "last_units_vs_normal", "estimates_6m",
    # C. billing behaviour
    "last_reading_estimated", "late_payments_6m",
    # D. next month's known context
    "target_month", "normal_temp_c", "tariff_hike_active",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES   # 22 features
TARGET = "next_bill"


def normal_temperatures(bills: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Usual temperature per region and calendar month, from TRAINING months only."""
    df = bills.merge(customers[["customer_id", "region"]], on="customer_id")
    df = df[pd.PeriodIndex(df["bill_month"], freq="M") <= TRAIN_END]
    df["target_month"] = df["bill_month"].str[5:7].astype(int)
    return (df.groupby(["region", "target_month"])["avg_temp_c"].mean()
              .round(1).rename("normal_temp_c").reset_index())


def build_model_table(bills: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = bills.sort_values(["customer_id", "bill_month"]).reset_index(drop=True)
    month = pd.PeriodIndex(df["bill_month"], freq="M")
    g = df.groupby("customer_id")

    out = pd.DataFrame({"customer_id": df["customer_id"]})
    out["predict_month"] = (month + 1).astype(str)          # the month we predict

    # B. Recent history: everything up to and including month M
    out["last_bill"] = df["bill_amount"]
    out["last_units"] = df["units_kwh"]
    out["bill_avg_3m"] = g["bill_amount"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    out["bill_avg_6m"] = g["bill_amount"].transform(lambda s: s.rolling(6, min_periods=3).mean())
    out["units_avg_3m"] = g["units_kwh"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    out["units_trend"] = g["units_kwh"].diff()
    out["bill_std_6m"] = g["bill_amount"].transform(lambda s: s.rolling(6, min_periods=3).std())
    # Same calendar month last year as the month we predict: M+1-12 = M-11
    out["bill_same_month_last_year"] = g["bill_amount"].shift(11)
    # How far last month's billed usage sits from the customer's normal (previous 6 months).
    # A low value after an estimated reading signals a catch-up bill is coming.
    prev_units_6m = g["units_kwh"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    out["last_units_vs_normal"] = df["units_kwh"] / prev_units_6m.replace(0, np.nan)
    out["history_months"] = g.cumcount() + 1

    # C. Billing behaviour. Bill M is paid during month M+1, so at the end of M we only
    # know whether bills up to M-1 were paid late.
    paid_late = (df["days_to_pay"] > LATE_AFTER_DAYS).astype(int)
    out["late_payments_6m"] = paid_late.groupby(df["customer_id"]).transform(
        lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0)
    estimated = (df["reading_type"] == "Estimated").astype(int)
    out["last_reading_estimated"] = estimated
    out["estimates_6m"] = estimated.groupby(df["customer_id"]).transform(
        lambda s: s.rolling(6, min_periods=1).sum())

    # D. Known context of the predicted month
    predict = month + 1
    out["target_month"] = predict.month
    out["tariff_hike_active"] = (predict >= TARIFF_HIKE_FROM).astype(int)

    # The answer: next month's actual bill (empty for the latest month = true future)
    out[TARGET] = g["bill_amount"].shift(-1)

    # A. Customer details
    out = out.merge(customers, on="customer_id", how="left")
    start = pd.PeriodIndex(out["connection_start"], freq="M")
    out["customer_age_months"] = (pd.PeriodIndex(out["predict_month"], freq="M").astype("int64")
                                  - start.astype("int64"))
    out = out.merge(normal_temperatures(bills, customers), on=["region", "target_month"], how="left")

    # Seasonal baseline for the high-bill flag (not a model input)
    out["flag_baseline"] = out[["bill_avg_6m", "bill_same_month_last_year"]].max(axis=1)

    # Eligibility and split
    out = out[out["history_months"] >= MIN_HISTORY_MONTHS].copy()
    pm = pd.PeriodIndex(out["predict_month"], freq="M")
    out["split"] = np.select(
        [pm <= TRAIN_END, (pm >= TEST_START) & (pm <= TEST_END), pm == FORECAST_MONTH],
        ["train", "test", "forecast"], default="drop")
    out = out[out["split"] != "drop"]
    out = out[(out["split"] == "forecast") | out[TARGET].notna()]
    return out.reset_index(drop=True)


def main() -> None:
    bills = pd.read_csv(PROCESSED_DIR / "bills_clean.csv")
    customers = pd.read_csv(PROCESSED_DIR / "customers_clean.csv")
    table = build_model_table(bills, customers)
    table.to_csv(PROCESSED_DIR / "model_table.csv", index=False)
    print(f"Model table: {len(table):,} rows, {len(FEATURES)} features")
    print(table["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
