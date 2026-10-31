"""
Train and predict (layer 4).

Models, all trained ONLY on the train months:
  - 3 baselines (no learning): last bill, 3-month average, same month last year
  - Ridge regression (simple, transparent)
  - LightGBM (main model) + two LightGBM quantile models for a 10th-90th percentile range

The test months (Sep-Dec 2025) are predicted but never used for training.
Finally, models are refit on train+test to forecast the real next month (Jan 2026).

Run from the project root:  python -m src.model.train
"""
import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.features.build import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

LGBM_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31,
                   "min_child_samples": 40, "subsample": 0.8, "subsample_freq": 1,
                   "colsample_bytree": 0.8, "random_state": 42, "verbose": -1}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def make_ridge():
    preprocess = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return make_pipeline(preprocess, Ridge(alpha=1.0))


def fit_models(train: pd.DataFrame) -> dict:
    X, y = prepare(train), train[TARGET]
    return {
        "ridge": make_ridge().fit(train[FEATURES], y),
        "lightgbm": LGBMRegressor(**LGBM_PARAMS).fit(X, y),
        "low": LGBMRegressor(objective="quantile", alpha=0.1, **LGBM_PARAMS).fit(X, y),
        "high": LGBMRegressor(objective="quantile", alpha=0.9, **LGBM_PARAMS).fit(X, y),
    }


def predict(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    out = df[["customer_id", "predict_month", "split", "connection_type", "region",
              "history_months", "last_bill", "bill_avg_6m", "bill_same_month_last_year",
              "flag_baseline", "last_reading_estimated", TARGET]].copy()
    out = out.rename(columns={TARGET: "actual_bill"})
    out["pred_last_bill"] = df["last_bill"]
    out["pred_avg_3m"] = df["bill_avg_3m"]
    out["pred_same_month_last_year"] = df["bill_same_month_last_year"].fillna(df["last_bill"])
    out["pred_ridge"] = models["ridge"].predict(df[FEATURES])
    out["pred_lightgbm"] = models["lightgbm"].predict(X).clip(min=0)
    low, high = models["low"].predict(X), models["high"].predict(X)
    # Quantile models are trained separately; keep the range ordered around the prediction
    out["low"] = pd.concat([pd.Series(low, index=out.index), out["pred_lightgbm"]], axis=1).min(axis=1).clip(lower=0)
    out["high"] = pd.concat([pd.Series(high, index=out.index), out["pred_lightgbm"]], axis=1).max(axis=1)
    return out


def main() -> None:
    table = pd.read_csv(PROCESSED_DIR / "model_table.csv")
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    forecast = table[table["split"] == "forecast"]

    # 1. Honest test: train on train months only, predict the held-out test months
    models = fit_models(train)
    test_pred = predict(models, test)

    # 2. Real forecast: refit on every known month, predict the next month
    final_models = fit_models(pd.concat([train, test]))
    forecast_pred = predict(final_models, forecast)

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    pd.concat([test_pred, forecast_pred]).to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    joblib.dump(models, MODEL_DIR / "models_test.joblib")
    joblib.dump(final_models, MODEL_DIR / "models_final.joblib")
    print(f"Trained on {len(train):,} rows; predicted {len(test_pred):,} test rows "
          f"and {len(forecast_pred):,} forecast rows")


if __name__ == "__main__":
    main()
