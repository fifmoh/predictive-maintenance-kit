from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parents[2]
st.set_page_config(page_title="Predictive maintenance kit", layout="wide")
st.title("Predictive maintenance with decision economics")
st.warning("Synthetic fixture — costs are illustrative units, never currency or client savings.")
pages = [
    "Dataset",
    "Degradation",
    "Predictions",
    "Threshold Economics",
    "Transfer Study",
    "Data/Model Cards",
]
page = st.sidebar.radio("Page", pages)
path = ROOT / "artifacts/runs/fixture-maintenance/result.json"
if page == "Threshold Economics":
    failure = st.select_slider("Failure cost", options=[5, 10, 20], value=10)
    downtime = st.select_slider("Downtime cost", options=[0, 1, 2], value=1)
    early = st.select_slider("Early-cycle cost", options=[0.0025, 0.005, 0.01], value=0.005)
    if path.exists():
        rows = json.loads(path.read_text())["sensitivity"]
        row = next(
            item
            for item in rows
            if item["failure_cost"] == failure
            and item["downtime_cost"] == downtime
            and item["early_cycle_cost"] == early
        )
        st.metric("Selected RUL threshold", row["threshold"])
        st.metric("Mean illustrative cost", row["mean_cost"])
else:
    st.write(
        "FD001: deterministic 70/15/15 complete run-to-failure engines. "
        "Official truncated test is used only for endpoint RUL metrics; "
        "policy cost is forbidden there."
    )
    st.write(
        "AI4I target: Machine failure; fixed stratified 70/15/15 split, "
        "validation calibration and thresholds, untouched test."
    )
