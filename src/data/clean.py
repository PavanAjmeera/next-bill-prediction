"""
Clean and validate the raw billing data (layer 2).

Steps, in order:
  1. Standardise text      - " URBAN", "semi urban" -> "Urban", "Semi-urban"
  2. Remove duplicates     - the same bill loaded twice
  3. Impossible readings   - negative units -> treated as missing
  4. Fill missing units    - from the customer's neighbouring months
  5. Validate              - stop loudly if any rule is broken
  6. Report                - what was found and fixed, for the README and dashboard

Run from the project root:  python -m src.data.clean
"""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

REGION_MAP = {"urban": "Urban", "semi urban": "Semi-urban", "semi-urban": "Semi-urban", "rural": "Rural"}


def clean_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = customers.copy()
    raw_regions = df["region"].copy()
    df["region"] = df["region"].str.strip().str.lower().map(REGION_MAP)
    fixed = int((raw_regions != df["region"]).sum())
    assert df["region"].notna().all(), "Unknown region value"
    assert df["customer_id"].is_unique, "Duplicate customer IDs"
    return df, {"region_values_standardised": fixed}


def clean_bills(bills: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = bills.copy()
    report = {"rows_raw": len(df)}

    # One bill per customer per month: keep the first copy
    before = len(df)
    df = df.drop_duplicates(subset=["customer_id", "bill_month"], keep="first")
    report["duplicate_bills_removed"] = before - len(df)

    # Negative usage is impossible -> treat as missing
    negative = df["units_kwh"] < 0
    report["negative_readings_fixed"] = int(negative.sum())
    df.loc[negative, "units_kwh"] = np.nan

    # Fill missing units from the same customer's neighbouring months
    report["missing_units_filled"] = int(df["units_kwh"].isna().sum())
    df = df.sort_values(["customer_id", "bill_month"]).reset_index(drop=True)
    df["units_imputed"] = df["units_kwh"].isna().astype(int)
    df["units_kwh"] = df.groupby("customer_id")["units_kwh"].transform(
        lambda s: s.interpolate(limit_direction="both")
    ).round()

    report["rows_clean"] = len(df)
    return df, report


def validate(bills: pd.DataFrame, customers: pd.DataFrame) -> None:
    """Final gate: the pipeline stops if the clean data breaks any rule."""
    assert not bills.duplicated(["customer_id", "bill_month"]).any(), "Duplicate bills"
    assert bills["units_kwh"].notna().all() and (bills["units_kwh"] >= 0).all(), "Bad units"
    assert (bills["bill_amount"] > 0).all(), "Bill must be positive"
    assert bills["customer_id"].isin(customers["customer_id"]).all(), "Bill for unknown customer"
    assert set(bills["reading_type"]) <= {"Actual", "Estimated"}, "Unknown reading type"
    # Each customer must have consecutive months (features rely on this)
    months = pd.PeriodIndex(bills["bill_month"], freq="M").astype("int64")
    gaps = pd.Series(months).groupby(bills["customer_id"].values).diff().dropna()
    assert (gaps == 1).all(), "Missing months inside a customer's history"


def main() -> None:
    customers, c_report = clean_customers(pd.read_csv(RAW_DIR / "customers.csv"))
    bills, b_report = clean_bills(pd.read_csv(RAW_DIR / "bills.csv"))
    validate(bills, customers)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(PROCESSED_DIR / "customers_clean.csv", index=False)
    bills.to_csv(PROCESSED_DIR / "bills_clean.csv", index=False)
    report = {**c_report, **b_report}
    (PROCESSED_DIR / "quality_report.json").write_text(json.dumps(report, indent=2))
    print("Cleaning report:", report)


if __name__ == "__main__":
    main()
