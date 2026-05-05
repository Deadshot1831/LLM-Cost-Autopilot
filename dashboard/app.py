"""Streamlit dashboard for LLM Cost Autopilot.

Run:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from autopilot.db import (
    open_db,
    query_aggregate_costs,
    query_escalation_rate_over_time,
    query_recent,
    query_routing_distribution,
    query_verdict_distribution,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "autopilot.db"

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("LLM Cost Autopilot")

db_path = st.sidebar.text_input("Database path", value=str(DEFAULT_DB))
db_file = Path(db_path)

if not db_file.exists():
    st.warning(
        f"No database found at `{db_path}`.\n\n"
        "Run `uv run python scripts/load_test.py` to populate it, "
        "then refresh this page."
    )
    st.stop()

conn = open_db(db_file)
agg = query_aggregate_costs(conn)

# ---- Headline metric ----
st.header("Cost Reduction")
left, mid, right = st.columns(3)
left.metric("Total requests", f"{agg['total_requests']:,}")
mid.metric(
    "Final cost",
    f"${agg['final_cost_total']:.4f}",
    delta=f"-${agg['savings_total']:.4f} vs baseline",
    delta_color="inverse",
)
right.metric(
    "Savings",
    f"{agg['savings_pct']:.1f}%",
    help="Compared to running every request through gpt-4o.",
)

st.caption(
    f"Baseline (every request via gpt-4o): ${agg['baseline_cost_total']:.4f}"
)

# ---- Routing distribution ----
st.header("Routing distribution")
routing = pd.DataFrame(query_routing_distribution(conn))
if not routing.empty:
    chart = (
        alt.Chart(routing)
        .mark_arc(innerRadius=60)
        .encode(theta="count:Q", color="model:N", tooltip=["model", "count"])
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No routing data yet.")

# ---- Verdict distribution ----
st.header("Quality verdicts")
verdicts = pd.DataFrame(query_verdict_distribution(conn))
if not verdicts.empty:
    chart = (
        alt.Chart(verdicts)
        .mark_bar()
        .encode(
            x="verdict:N", y="count:Q", color="verdict:N",
            tooltip=["verdict", "count"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No verdicts yet.")

# ---- Escalation rate over time ----
st.header("Escalation rate over time")
esc = pd.DataFrame(query_escalation_rate_over_time(conn))
if not esc.empty:
    chart = (
        alt.Chart(esc)
        .mark_line(point=True)
        .encode(
            x="day:T",
            y=alt.Y(
                "escalation_rate:Q",
                title="escalation rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=["day", "escalation_rate", "n"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No escalation data yet.")

# ---- Recent requests ----
st.header("Recent requests")
recent = pd.DataFrame(query_recent(conn, limit=50))
if not recent.empty:
    display = recent[
        [
            "timestamp", "tier", "candidate_model", "verdict",
            "escalated", "final_model", "final_cost", "baseline_cost",
            "prompt_preview",
        ]
    ]
    st.dataframe(display, use_container_width=True)
else:
    st.info("No requests yet.")
