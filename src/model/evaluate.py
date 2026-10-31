"""
Evaluate, explain and flag (layer 5).

Outputs (all in outputs/):
  metrics.json         - MAE, RMSE, WAPE per method on the test months, range coverage, flag quality
  segments.csv         - LightGBM error by connection type, history length, bill range, reading type
  worst_cases.csv      - the 10 biggest test errors, for failure analysis
  feature_importance.csv - average SHAP influence of each feature
  priority_list.csv    - the actionable view for the forecast month, highest risk first

Run from the project root:  python -m src.model.evaluate
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import HIGH_BILL_MULTIPLIER, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR
from src.features.build import FEATURES
from src.model.train import prepare

METHODS = {
    "Last bill (baseline)": "pred_last_bill",
    "3-month average (baseline)": "pred_avg_3m",
    "Same month last year (baseline)": "pred_same_month_last_year",
    "Ridge regression": "pred_ridge",
    "LightGBM": "pred_lightgbm",
}

READABLE = {
    "last_bill": "last bill", "last_units": "last month's usage", "bill_avg_3m": "3-month average bill",
    "bill_avg_6m": "6-month average bill", "units_avg_3m": "3-month average usage",
    "units_trend": "usage trend", "bill_std_6m": "bill variability",
    "bill_same_month_last_year": "same month last year", "last_units_vs_normal": "last usage vs normal",
    "estimates_6m": "recent estimated readings", "last_reading_estimated": "last reading estimated",
    "late_payments_6m": "recent late payments", "target_month": "season (month)",
    "normal_temp_c": "expected temperature", "tariff_hike_active": "tariff increase",
    "sanctioned_load_kw": "sanctioned load", "household_size": "household size", "has_ac": "AC ownership",
    "customer_age_months": "customer age", "connection_type": "connection type", "region": "region",
    "meter_type": "meter type",
}


def error_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    err = actual - predicted
    return {
        "MAE": round(float(err.abs().mean()), 1),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 1),
        "WAPE": round(float(err.abs().sum() / actual.abs().sum()), 4),
    }


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Alert = predicted bill above 1.3 x baseline. Watch = upper end of the range above it."""
    df = df.copy()
    threshold = HIGH_BILL_MULTIPLIER * df["flag_baseline"]
    df["pct_above_baseline"] = (df["pred_lightgbm"] / df["flag_baseline"] - 1).round(3)
    df["flag"] = np.select([df["pred_lightgbm"] > threshold, df["high"] > threshold],
                           ["Alert", "Watch"], default="Normal")
    df["range_width_pct"] = ((df["high"] - df["low"]) / df["pred_lightgbm"]).round(3)
    df["confidence"] = np.select([df["history_months"] < 6, df["range_width_pct"] > 0.5],
                                 ["Low", "Low"], default="High")
    return df


def flag_quality(test: pd.DataFrame) -> dict:
    actual_high = test["actual_bill"] > HIGH_BILL_MULTIPLIER * test["flag_baseline"]
    result = {"actual_high_bills": int(actual_high.sum()),
              "base_rate": round(float(actual_high.mean()), 4)}
    for name, flagged in {"alert_only": test["flag"] == "Alert",
                          "alert_or_watch": test["flag"].isin(["Alert", "Watch"])}.items():
        hits = int((flagged & actual_high).sum())
        result[name] = {"flagged": int(flagged.sum()),
                        "precision": round(hits / max(int(flagged.sum()), 1), 3),
                        "recall": round(hits / max(int(actual_high.sum()), 1), 3)}
    return result


def segment_errors(test: pd.DataFrame) -> pd.DataFrame:
    test = test.copy()
    test["history_length"] = pd.cut(test["history_months"], [0, 6, 12, 99],
                                     labels=["3-6 months", "7-12 months", "13+ months"])
    test["bill_range"] = pd.cut(test["last_bill"], [0, 1000, 3000, np.inf],
                                labels=["under 1,000", "1,000-3,000", "above 3,000"])
    test["reading"] = test["last_reading_estimated"].map({0: "last actual", 1: "last estimated"})
    rows = []
    for segment in ["connection_type", "history_length", "bill_range", "reading"]:
        for value, grp in test.groupby(segment, observed=True):
            rows.append({"segment": segment, "group": str(value), "rows": len(grp),
                         **error_metrics(grp["actual_bill"], grp["pred_lightgbm"])})
    return pd.DataFrame(rows)


def explain(model, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP: how much each feature pushed each prediction up or down (influence, not cause)."""
    X = prepare(table)
    # LightGBM computes exact TreeSHAP values itself (last column = the average prediction)
    values = model.predict(X, pred_contrib=True)[:, :-1]
    contrib = pd.DataFrame(values, columns=FEATURES, index=table.index)
    importance = (contrib.abs().mean().sort_values(ascending=False)
                  .rename("mean_abs_shap_rupees").round(1).reset_index()
                  .rename(columns={"index": "feature"}))

    def top3(row):
        # For a high-bill review, what matters is what pushed the bill UP
        up = row[row > 0].sort_values(ascending=False)
        top = up.iloc[:3] if len(up) else row.reindex(row.abs().sort_values(ascending=False).index[:3])
        return "; ".join(f"{READABLE.get(f, f)} {'+' if v > 0 else '-'}₹{abs(v):,.0f}" for f, v in top.items())

    drivers = contrib.apply(top3, axis=1).rename("top_drivers")
    return importance, drivers


def main() -> None:
    preds = add_flags(pd.read_csv(OUTPUT_DIR / "predictions.csv"))
    test = preds[preds["split"] == "test"]
    forecast = preds[preds["split"] == "forecast"].copy()

    # 1. Accuracy against baselines
    metrics = {"test_rows": len(test),
               "test_months": sorted(test["predict_month"].unique().tolist()),
               "methods": {name: error_metrics(test["actual_bill"], test[col]) for name, col in METHODS.items()}}
    inside = (test["actual_bill"] >= test["low"]) & (test["actual_bill"] <= test["high"])
    metrics["range_coverage"] = round(float(inside.mean()), 3)   # target about 0.80
    metrics["flag_quality"] = flag_quality(test)

    # 2. Where the model is weaker
    segments = segment_errors(test)

    # 3. Failure analysis: the biggest misses
    worst = test.assign(abs_error=(test["actual_bill"] - test["pred_lightgbm"]).abs()).nlargest(10, "abs_error")

    # 4. Explanations for the forecast month (final model)
    table = pd.read_csv(PROCESSED_DIR / "model_table.csv")
    final_model = joblib.load(MODEL_DIR / "models_final.joblib")["lightgbm"]
    future_rows = table[table["split"] == "forecast"]
    importance, drivers = explain(final_model, future_rows)
    forecast["top_drivers"] = drivers.values

    # 5. Actionable priority list: Alerts first, then Watch, then by % above baseline
    order = {"Alert": 0, "Watch": 1, "Normal": 2}
    priority = (forecast.assign(_o=forecast["flag"].map(order))
                .sort_values(["_o", "pct_above_baseline"], ascending=[True, False])
                [["customer_id", "predict_month", "flag", "pred_lightgbm", "low", "high",
                  "last_bill", "bill_avg_6m", "bill_same_month_last_year", "flag_baseline",
                  "pct_above_baseline", "confidence", "top_drivers", "connection_type", "region"]]
                .rename(columns={"pred_lightgbm": "predicted_bill"}).round(2))

    metrics["forecast_month"] = str(forecast["predict_month"].iloc[0])
    metrics["forecast_flags"] = priority["flag"].value_counts().to_dict()

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    segments.to_csv(OUTPUT_DIR / "segments.csv", index=False)
    worst.to_csv(OUTPUT_DIR / "worst_cases.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    priority.to_csv(OUTPUT_DIR / "priority_list.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_results.csv", index=False)

    best = min(metrics["methods"].items(), key=lambda kv: kv[1]["WAPE"])
    print(f"Best method: {best[0]} (WAPE {best[1]['WAPE']:.1%}); range coverage {metrics['range_coverage']:.0%}")
    print("Flag quality:", metrics["flag_quality"])
    print(f"Forecast {metrics['forecast_month']}: {metrics['forecast_flags']}")


if __name__ == "__main__":
    main()
