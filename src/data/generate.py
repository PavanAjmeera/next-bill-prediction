"""
Generate a synthetic electricity billing dataset (layer 1).

Two files, like a real utility:
  data/raw/customers.csv  - one row per customer
  data/raw/bills.csv      - one row per customer per month (Jan 2024 - Dec 2025)

All billing rules are written down below (the brief requires this).
The tariff is ILLUSTRATIVE, inspired by typical Indian slab tariffs,
not copied from any real electricity company.

Run from the project root:  python -m src.data.generate
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42                 # same seed -> same data every time (reproducible)
N_CUSTOMERS = 2000
MONTHS = pd.period_range("2024-01", "2025-12", freq="M")
RAW_DIR = Path("data/raw")

# ---------------- Billing rules (documented) ----------------
DOMESTIC_SLABS = [(100, 3.00), (100, 4.50), (200, 6.50), (np.inf, 8.00)]  # (units in slab, rate)
COMMERCIAL_RATE = 8.50
FIXED_PER_KW = {"Domestic": 50, "Commercial": 150}
DUTY_RATE = 0.06            # electricity duty on the energy charge
LATE_FEE_RATE = 0.015       # 1.5% of the previous bill if it was paid late
LATE_AFTER_DAYS = 20        # paying after 20 days counts as late
TARIFF_HIKE_FROM = pd.Period("2025-04", freq="M")
TARIFF_HIKE = 1.05          # rates rise 5% from April 2025

# ---------------- Climate: normal temperature (deg C) by month ----------------
NORMAL_TEMP = {1: 21, 2: 24, 3: 28, 4: 32, 5: 35, 6: 33,
               7: 29, 8: 28, 9: 28, 10: 27, 11: 24, 12: 21}
REGION_TEMP_SHIFT = {"Urban": 1.0, "Semi-urban": 0.0, "Rural": -1.0}


def energy_charge(units: float, connection: str, month: pd.Period) -> float:
    """Slab-wise energy charge: higher usage is billed at higher rates."""
    hike = TARIFF_HIKE if month >= TARIFF_HIKE_FROM else 1.0
    if connection == "Commercial":
        return units * COMMERCIAL_RATE * hike
    charge, remaining = 0.0, units
    for size, rate in DOMESTIC_SLABS:
        used = min(remaining, size)
        charge += used * rate * hike
        remaining -= used
        if remaining <= 0:
            break
    return charge


def make_customers(rng: np.random.Generator) -> pd.DataFrame:
    n = N_CUSTOMERS
    connection = rng.choice(["Domestic", "Commercial"], n, p=[0.85, 0.15])
    region = rng.choice(["Urban", "Semi-urban", "Rural"], n, p=[0.5, 0.3, 0.2])
    load_kw = np.where(connection == "Domestic",
                       rng.choice([1, 2, 3, 5], n, p=[0.2, 0.35, 0.3, 0.15]),
                       rng.choice([5, 10, 15, 20], n, p=[0.4, 0.3, 0.2, 0.1]))
    household = np.where(connection == "Domestic", rng.integers(1, 7, n), 0)
    ac_prob = np.select([region == "Urban", region == "Semi-urban"], [0.55, 0.35], 0.15)
    has_ac = rng.binomial(1, np.where(connection == "Commercial", 0.7, ac_prob))
    meter = rng.choice(["Smart", "Analog"], n, p=[0.6, 0.4])
    # 85% were customers before Jan 2024; 15% join during the period (short history)
    start = np.where(
        rng.random(n) < 0.85,
        [str(pd.Period("2018-01", "M") + int(k)) for k in rng.integers(0, 72, n)],
        [str(MONTHS[int(k)]) for k in rng.integers(1, 18, n)],
    )
    return pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(n)],
        "connection_type": connection,
        "region": region,
        "sanctioned_load_kw": load_kw,
        "household_size": household,
        "has_ac": has_ac,
        "meter_type": meter,
        "connection_start": start,
    })


def season_factor(month_num: int, has_ac: int) -> float:
    """Summer peak (Apr-Jun), monsoon medium, winter low; AC owners swing much more."""
    base = {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.15, 5: 1.30, 6: 1.20,
            7: 1.05, 8: 1.00, 9: 1.00, 10: 0.95, 11: 0.90, 12: 0.85}[month_num]
    return 1 + (base - 1) * (1.5 if has_ac else 0.7)


def make_bills(customers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for c in customers.itertuples(index=False):
        if c.connection_type == "Domestic":
            base_units = 60 + 30 * c.household_size + 25 * c.sanctioned_load_kw
        else:
            base_units = 80 * c.sanctioned_load_kw
        base_units *= rng.uniform(0.75, 1.25)          # every customer is a bit different
        growth = rng.normal(0.003, 0.004)               # slow drift per month

        event = rng.choice(["none", "new_appliance", "vacant"], p=[0.88, 0.08, 0.04])
        event_month = int(rng.integers(6, 22))
        est_prob = 0.12 if c.meter_type == "Analog" else 0.02

        carry_units = 0.0           # usage not billed because of an estimated reading
        prev_bill, prev_paid_late = None, False
        history_units = []

        for t, month in enumerate(MONTHS):
            if month < pd.Period(c.connection_start, "M"):
                continue
            temp = NORMAL_TEMP[month.month] + REGION_TEMP_SHIFT[c.region] + rng.normal(0, 1.2)
            units = base_units * (1 + growth) ** t * season_factor(month.month, c.has_ac)
            units *= 1 + 0.03 * (temp - NORMAL_TEMP[month.month])  # hotter than usual -> more
            if event == "new_appliance" and t >= event_month:
                units *= 1.45
            if event == "vacant" and event_month <= t < event_month + 3:
                units *= 0.08
            units = max(units * rng.normal(1, 0.06), 0)

            # Estimated reading: billed at the recent average; the gap is billed next month
            estimated = len(history_units) >= 3 and rng.random() < est_prob
            if estimated:
                billed_units = float(np.mean(history_units[-3:])) * 0.8
                carry_units += units - billed_units
            else:
                billed_units = units + carry_units       # catch-up after an estimate
                carry_units = 0.0
            billed_units = max(round(billed_units), 0)
            history_units.append(billed_units)

            e_charge = energy_charge(billed_units, c.connection_type, month)
            fixed = FIXED_PER_KW[c.connection_type] * c.sanctioned_load_kw
            late_fee = LATE_FEE_RATE * prev_bill if prev_paid_late else 0.0
            tax = DUTY_RATE * e_charge
            bill = e_charge + fixed + late_fee + tax

            days_to_pay = int(np.clip(rng.gamma(2.0, 7.0), 1, 60))
            rows.append({
                "customer_id": c.customer_id,
                "bill_month": str(month),
                "billing_days": month.days_in_month,
                "units_kwh": billed_units,
                "avg_temp_c": round(temp, 1),
                "reading_type": "Estimated" if estimated else "Actual",
                "energy_charge": round(e_charge, 2),
                "fixed_charge": float(fixed),
                "late_fee": round(late_fee, 2),
                "tax": round(tax, 2),
                "bill_amount": round(bill, 2),
                "days_to_pay": days_to_pay,
            })
            prev_bill, prev_paid_late = bill, days_to_pay > LATE_AFTER_DAYS
    return pd.DataFrame(rows)


def add_realistic_mess(customers, bills, rng):
    """Real data is messy; the cleaning layer must handle all of this."""
    customers, bills = customers.copy(), bills.copy()
    # Region typed differently by different staff
    idx = rng.choice(len(customers), 60, replace=False)
    customers.loc[idx, "region"] = rng.choice(["urban", "URBAN ", "semi urban", " Rural"], 60)
    # Missing meter readings
    bills.loc[rng.choice(len(bills), int(len(bills) * 0.01), replace=False), "units_kwh"] = np.nan
    # Impossible readings
    bills.loc[rng.choice(len(bills), 25, replace=False), "units_kwh"] = -1
    # Duplicate bill rows (same bill loaded twice)
    bills = pd.concat([bills, bills.sample(150, random_state=SEED)], ignore_index=True)
    bills = bills.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return customers, bills


def main() -> None:
    rng = np.random.default_rng(SEED)
    customers = make_customers(rng)
    bills = make_bills(customers, rng)
    customers, bills = add_realistic_mess(customers, bills, rng)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(RAW_DIR / "customers.csv", index=False)
    bills.to_csv(RAW_DIR / "bills.csv", index=False)
    print(f"customers.csv: {len(customers):,} rows")
    print(f"bills.csv:     {len(bills):,} rows ({bills['bill_month'].min()} to {bills['bill_month'].max()})")


if __name__ == "__main__":
    main()
