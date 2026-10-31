"""Shared settings for every layer: paths, cutoff dates and the high-bill rule."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

# Chronological split: predictions for months up to TRAIN_END train the model,
# months from TEST_START to TEST_END are held out for the final test.
TRAIN_END = pd.Period("2025-08", freq="M")
TEST_START = pd.Period("2025-09", freq="M")
TEST_END = pd.Period("2025-12", freq="M")
FORECAST_MONTH = pd.Period("2026-01", freq="M")   # the real "next bill" we forecast

# High-bill rule: next bill above 1.3 x the customer's own baseline, where the
# baseline is the HIGHER of their last-6-bill average and the same month last year.
# Using last year stops normal summer rises from being flagged as "unusual".
HIGH_BILL_MULTIPLIER = 1.3

MIN_HISTORY_MONTHS = 3      # a customer needs 3+ bills to get a prediction
