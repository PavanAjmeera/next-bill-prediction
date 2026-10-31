"""
High-bill monitoring dashboard (layer 6).

Run from the project root (after the pipeline):  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
OUT, PROC = ROOT / "outputs", ROOT / "data" / "processed"

st.set_page_config(page_title="Next Bill Prediction", layout="wide")


@st.cache_data
def load():
    return {
        "priority": pd.read_csv(OUT / "priority_list.csv"),
        "metrics": json.loads((OUT / "metrics.json").read_text()),
        "segments": pd.read_csv(OUT / "segments.csv"),
        "test": pd.read_csv(OUT / "test_results.csv"),
        "importance": pd.read_csv(OUT / "feature_importance.csv"),
        "worst": pd.read_csv(OUT / "worst_cases.csv"),
        "bills": pd.read_csv(PROC / "bills_clean.csv"),
        "customers": pd.read_csv(PROC / "customers_clean.csv"),
    }


if not (OUT / "priority_list.csv").exists():
    st.error("No outputs yet. Run the pipeline first:  python run_pipeline.py")
    st.stop()

d = load()
m = d["metrics"]
st.title("Next Bill Amount Prediction")
st.caption(f"Forecast for {m['forecast_month']} · model tested on {', '.join(m['test_months'])} "
           "(months it never saw during training)")

tab_alerts, tab_accuracy, tab_explore = st.tabs(["High-bill alerts", "Model accuracy", "Billing patterns"])

# ---------------- Tab 1: actionable view (FR-06, FR-08) ----------------
with tab_alerts:
    pr = d["priority"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers forecast", f"{len(pr):,}")
    c2.metric("Alert", int((pr["flag"] == "Alert").sum()), help="Predicted bill above 1.3 × the customer's baseline")
    c3.metric("Watch", int((pr["flag"] == "Watch").sum()), help="Upper end of the range above 1.3 × baseline")
    c4.metric("Model WAPE", f"{m['methods']['LightGBM']['WAPE']:.1%}")

    st.markdown("**Baseline** = the higher of the customer's last-6-bill average and the same month last "
                "year, so normal seasonal rises are not flagged.")
    flags = st.multiselect("Show", ["Alert", "Watch", "Normal"], default=["Alert", "Watch"])
    view = pr[pr["flag"].isin(flags)]
    st.dataframe(
        view[["customer_id", "flag", "predicted_bill", "low", "high", "flag_baseline",
              "pct_above_baseline", "confidence", "top_drivers", "connection_type", "region"]],
        width="stretch", hide_index=True,
        column_config={
            "predicted_bill": st.column_config.NumberColumn("Predicted ₹", format="%.0f"),
            "low": st.column_config.NumberColumn("Low ₹", format="%.0f"),
            "high": st.column_config.NumberColumn("High ₹", format="%.0f"),
            "flag_baseline": st.column_config.NumberColumn("Baseline ₹", format="%.0f"),
            "pct_above_baseline": st.column_config.NumberColumn("vs baseline", format="percent"),
            "top_drivers": "Main drivers (influence, not proven cause)",
        })

    st.subheader("Customer detail")
    options = view["customer_id"].tolist() or pr["customer_id"].tolist()
    cid = st.selectbox("Customer", options)
    row = pr[pr["customer_id"] == cid].iloc[0]
    hist = d["bills"][d["bills"]["customer_id"] == cid].sort_values("bill_month")
    fig = go.Figure()
    fig.add_bar(x=hist["bill_month"], y=hist["bill_amount"], name="Past bills",
                marker_color=["#E07A2F" if r == "Estimated" else "#2F6FB0" for r in hist["reading_type"]])
    fig.add_scatter(x=[row["predict_month"]], y=[row["predicted_bill"]], mode="markers", name="Prediction",
                    marker={"size": 14, "color": "#14213D"},
                    error_y={"type": "data", "symmetric": False,
                             "array": [row["high"] - row["predicted_bill"]],
                             "arrayminus": [row["predicted_bill"] - row["low"]]})
    fig.add_hline(y=1.3 * row["flag_baseline"], line_dash="dash", annotation_text="High-bill threshold")
    fig.update_layout(height=380, yaxis_title="Bill (₹)", legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Orange bars = estimated meter readings. The error bar shows the 10th–90th percentile range.")
    st.write(f"**{row['flag']}** · predicted ₹{row['predicted_bill']:,.0f} "
             f"(range ₹{row['low']:,.0f}–₹{row['high']:,.0f}) · confidence: {row['confidence']} · "
             f"drivers: {row['top_drivers']}")

# ---------------- Tab 2: accuracy (FR-04, FR-07) ----------------
with tab_accuracy:
    methods = pd.DataFrame(m["methods"]).T.reset_index().rename(columns={"index": "Method"})
    st.subheader("Model vs simple baselines (test months)")
    st.dataframe(methods.style.format({"MAE": "₹{:,.0f}", "RMSE": "₹{:,.0f}", "WAPE": "{:.1%}"}),
                 width="stretch", hide_index=True)
    st.write(f"Prediction range coverage: **{m['range_coverage']:.0%}** of actual bills fell inside the "
             "10th–90th percentile range (target about 80%).")

    fq = m["flag_quality"]
    st.subheader("High-bill flag quality")
    st.write(f"{fq['actual_high_bills']} test bills were genuinely unusual ({fq['base_rate']:.1%} of all bills).")
    st.dataframe(pd.DataFrame({
        "Flag": ["Alert only", "Alert or Watch"],
        "Flagged": [fq["alert_only"]["flagged"], fq["alert_or_watch"]["flagged"]],
        "Precision": [fq["alert_only"]["precision"], fq["alert_or_watch"]["precision"]],
        "Recall": [fq["alert_only"]["recall"], fq["alert_or_watch"]["recall"]],
    }).style.format({"Precision": "{:.0%}", "Recall": "{:.0%}"}), width="stretch", hide_index=True)

    st.subheader("Where the model is weaker")
    seg = d["segments"]
    st.plotly_chart(px.bar(seg, x="group", y="WAPE", color="segment", text_auto=".1%",
                           labels={"group": ""}).update_layout(height=380, yaxis_tickformat=".0%"),
                    width="stretch")

    t = d["test"]
    st.subheader("Error distribution")
    st.plotly_chart(px.histogram(t.assign(error=t["actual_bill"] - t["pred_lightgbm"]), x="error", nbins=80,
                                 labels={"error": "Actual − predicted (₹)"}).update_layout(height=320),
                    width="stretch")
    st.subheader("Biggest misses")
    st.dataframe(d["worst"][["customer_id", "predict_month", "actual_bill", "pred_lightgbm",
                             "last_reading_estimated", "connection_type"]].round(0),
                 width="stretch", hide_index=True)

# ---------------- Tab 3: patterns and cost drivers (FR-01, FR-02) ----------------
with tab_explore:
    b = d["bills"].merge(d["customers"], on="customer_id")
    b["month"] = b["bill_month"].str[5:7]
    st.subheader("Seasonality: median bill by calendar month")
    season = b.groupby(["month", "has_ac"])["bill_amount"].median().reset_index()
    season["AC"] = season["has_ac"].map({0: "No AC", 1: "Has AC"})
    st.plotly_chart(px.line(season, x="month", y="bill_amount", color="AC", markers=True,
                            labels={"bill_amount": "Median bill (₹)", "month": "Month"}).update_layout(height=360),
                    width="stretch")
    st.subheader("Main cost drivers (average SHAP influence on the prediction, ₹)")
    imp = d["importance"].head(10).iloc[::-1]
    st.plotly_chart(px.bar(imp, x="mean_abs_shap_rupees", y="feature", orientation="h",
                           labels={"mean_abs_shap_rupees": "Average influence (₹)", "feature": ""})
                    .update_layout(height=400), width="stretch")
    st.caption("Influence shows what the model relies on, not proven billing causes.")