"""Streamlit dashboard for LLM Cost Autopilot.

Run:
    uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from autopilot.classifier import ComplexityClassifier
from autopilot.db import (
    open_db,
    query_aggregate_costs,
    query_escalation_rate_over_time,
    query_recent,
    query_routing_distribution,
    query_verdict_distribution,
)
from autopilot.features import extract_features
from autopilot.registry import load_registry
from autopilot.routing import load_routing_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "autopilot.db"
DEFAULT_CLF = ROOT / "models" / "classifier.joblib"
DEFAULT_REG = ROOT / "config" / "models.yaml"
DEFAULT_ROUTING = ROOT / "config" / "routing.yaml"


# ---------- caching helpers ----------

@st.cache_resource
def _load_classifier(path: str):
    return ComplexityClassifier.load(Path(path))


@st.cache_resource
def _load_registry(path: str):
    return load_registry(Path(path))


@st.cache_resource
def _load_routing(path: str):
    return load_routing_config(Path(path))


@st.cache_data(ttl=5)
def _agg_costs(db_path: str) -> dict:
    return query_aggregate_costs(open_db(db_path))


@st.cache_data(ttl=5)
def _routing_dist(db_path: str) -> list[dict]:
    return query_routing_distribution(open_db(db_path))


@st.cache_data(ttl=5)
def _verdict_dist(db_path: str) -> list[dict]:
    return query_verdict_distribution(open_db(db_path))


@st.cache_data(ttl=5)
def _escalation_over_time(db_path: str) -> list[dict]:
    return query_escalation_rate_over_time(open_db(db_path))


@st.cache_data(ttl=5)
def _all_requests(db_path: str) -> pd.DataFrame:
    return pd.DataFrame(query_recent(open_db(db_path), limit=10_000))


# ---------- page setup ----------

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")

st.title("LLM Cost Autopilot")
st.markdown(
    "A routing layer that sends each LLM request to the **cheapest model that can handle it**, "
    "then verifies the cheap model's answer against a reference model and **auto-escalates on disagreement**. "
    "This dashboard shows what it's doing and how much it's saving."
)

# Sidebar
st.sidebar.header("Configuration")
db_path = st.sidebar.text_input("Database path", value=str(DEFAULT_DB))
classifier_path = st.sidebar.text_input("Classifier path", value=str(DEFAULT_CLF))
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(
    "Data refreshes automatically every 5 seconds. "
    "Hit Refresh to bust the cache immediately."
)

db_file = Path(db_path)
if not db_file.exists():
    st.warning(
        f"No database found at `{db_path}`. "
        "Run `uv run python scripts/simulate_load.py -n 500` to populate it, "
        "then refresh this page."
    )
    st.stop()

agg = _agg_costs(str(db_file))

# ---------- headline KPIs ----------

st.header("Cost reduction")
st.caption(
    "**Baseline** = what it would have cost if every request had been sent to `gpt-4o` (the most expensive model in the registry). "
    "**Final** = what we actually spent after routing cheap requests to cheaper models. "
    "**Savings %** is the headline number."
)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total requests", f"{agg['total_requests']:,}")
k2.metric("Baseline cost", f"${agg['baseline_cost_total']:.4f}")
k3.metric(
    "Final cost",
    f"${agg['final_cost_total']:.4f}",
    delta=f"-${agg['savings_total']:.4f}",
    delta_color="inverse",
)
k4.metric(
    "Savings",
    f"{agg['savings_pct']:.1f}%",
    help="Compared to running every request through gpt-4o.",
)

# ---------- tabs ----------

tab_overview, tab_try, tab_drilldown = st.tabs(
    ["Overview", "Try it yourself", "Per-request drill-down"]
)

# =================== TAB 1: OVERVIEW ===================
with tab_overview:
    # --- Cumulative cost chart ---
    st.subheader("Cumulative cost: baseline vs final")
    st.caption(
        "Each line is the running total of dollars spent up to request N. "
        "The gap between the two is the money saved by routing. "
        "If the lines stay close, the classifier is routing everything to the expensive model — "
        "if they diverge sharply, the cheap routing is paying off."
    )
    all_df = _all_requests(str(db_file))
    if not all_df.empty:
        chrono = all_df.iloc[::-1].reset_index(drop=True).copy()
        chrono["request_n"] = chrono.index + 1
        chrono["cum_final"] = chrono["final_cost"].cumsum()
        chrono["cum_baseline"] = chrono["baseline_cost"].cumsum()
        long = pd.melt(
            chrono[["request_n", "cum_final", "cum_baseline"]],
            id_vars="request_n",
            var_name="series",
            value_name="cumulative_cost",
        )
        long["series"] = long["series"].map(
            {"cum_final": "Final (routed)", "cum_baseline": "Baseline (all gpt-4o)"}
        )
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("request_n:Q", title="Request number"),
                y=alt.Y("cumulative_cost:Q", title="Cumulative cost (USD)"),
                color=alt.Color("series:N", title=""),
                tooltip=["request_n", "series", alt.Tooltip("cumulative_cost:Q", format="$.4f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("No requests yet.")

    # --- Routing + verdict side by side ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Routing distribution")
        st.caption(
            "Which model actually handled each request. "
            "A healthy distribution sends most simple prompts to cheap models "
            "and saves the expensive model for genuinely hard ones."
        )
        routing_df = pd.DataFrame(_routing_dist(str(db_file)))
        if not routing_df.empty:
            chart = (
                alt.Chart(routing_df)
                .mark_arc(innerRadius=60)
                .encode(
                    theta="count:Q",
                    color="model:N",
                    tooltip=["model", "count"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")

    with col_b:
        st.subheader("Quality verdicts")
        st.caption(
            "**PASS** = cheap model agreed with the reference. "
            "**FAIL** = disagreed → the answer was auto-escalated to the reference. "
            "**SKIP** = no verification needed (the cheap model *was* the reference)."
        )
        verdicts_df = pd.DataFrame(_verdict_dist(str(db_file)))
        if not verdicts_df.empty:
            colour_scale = alt.Scale(
                domain=["pass", "fail", "skip"],
                range=["#2ca02c", "#d62728", "#7f7f7f"],
            )
            chart = (
                alt.Chart(verdicts_df)
                .mark_bar()
                .encode(
                    x=alt.X("verdict:N", title="Verdict"),
                    y=alt.Y("count:Q", title="Requests"),
                    color=alt.Color("verdict:N", scale=colour_scale, legend=None),
                    tooltip=["verdict", "count"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, width="stretch")

    # --- Escalation over time ---
    st.subheader("Escalation rate over time")
    st.caption(
        "Fraction of requests per day where the cheap model's answer was rejected. "
        "Trending **down** means the classifier is getting better at routing; "
        "trending **up** means it's getting overconfident with the cheap models."
    )
    esc_df = pd.DataFrame(_escalation_over_time(str(db_file)))
    if not esc_df.empty:
        chart = (
            alt.Chart(esc_df)
            .mark_line(point=True)
            .encode(
                x="day:T",
                y=alt.Y(
                    "escalation_rate:Q",
                    title="Escalation rate",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=["day", alt.Tooltip("escalation_rate:Q", format=".1%"), "n"],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.info("Not enough days of data yet (need at least 2 distinct days).")


# =================== TAB 2: TRY IT YOURSELF ===================
with tab_try:
    st.subheader("Predict the routing for a new prompt")
    st.markdown(
        "Type a prompt below. The classifier will predict the complexity tier "
        "(without making any API call), tell you which model the current routing "
        "config would send it to, and estimate the cost vs running it through `gpt-4o`."
    )

    try:
        clf = _load_classifier(classifier_path)
        registry = _load_registry(str(DEFAULT_REG))
        routing = _load_routing(str(DEFAULT_ROUTING))
    except Exception as e:
        st.error(f"Could not load classifier/registry/routing: {e}")
        st.stop()

    default_prompt = "What is the capital of Japan?"
    prompt_input = st.text_area(
        "Prompt", value=default_prompt, height=120,
        placeholder="Try: 'Translate hello to French' or 'Design a sharding strategy for Postgres'",
    )

    example_cols = st.columns(3)
    with example_cols[0]:
        if st.button("Try: simple math", width="stretch"):
            prompt_input = "What is 47 times 13?"
            st.session_state["_try_prompt"] = prompt_input
    with example_cols[1]:
        if st.button("Try: summarize", width="stretch"):
            prompt_input = "Summarize the plot of Romeo and Juliet in two sentences."
            st.session_state["_try_prompt"] = prompt_input
    with example_cols[2]:
        if st.button("Try: hard reasoning", width="stretch"):
            prompt_input = (
                "Compare and analyze the trade-offs between event sourcing and CRUD "
                "for a fintech ledger, including failure modes and operational complexity."
            )
            st.session_state["_try_prompt"] = prompt_input
    prompt_input = st.session_state.get("_try_prompt", prompt_input)

    if prompt_input and prompt_input.strip():
        tier, confidence = clf.predict_with_confidence(prompt_input)
        model_id = routing[tier]
        cfg = registry.get(model_id)
        baseline_cfg = registry.get("gpt-4o")
        features = extract_features(prompt_input)

        st.markdown("---")
        col_pred, col_cost = st.columns([1, 1])

        with col_pred:
            st.markdown("##### Classifier prediction")
            st.markdown(
                f"**Tier:** `{tier.value}`  \n"
                f"**Confidence:** {confidence:.1%}  \n"
                f"**Would route to:** `{model_id}` (provider: `{cfg.provider}`, "
                f"quality tier: `{cfg.quality_tier.value}`)"
            )
            st.caption(
                "Confidence below ~50% means the classifier is unsure. "
                "Those prompts are the most likely to get escalated by the verifier."
            )

        with col_cost:
            st.markdown("##### Estimated cost (1k in / 1k out)")
            routed_in = cfg.input_cost_per_1k * 1
            routed_out = cfg.output_cost_per_1k * 1
            baseline_in = baseline_cfg.input_cost_per_1k * 1
            baseline_out = baseline_cfg.output_cost_per_1k * 1
            routed_total = routed_in + routed_out
            baseline_total = baseline_in + baseline_out
            saved_pct = (
                (baseline_total - routed_total) / baseline_total * 100
                if baseline_total > 0 else 0.0
            )
            st.markdown(
                f"**Routed (`{model_id}`):** ${routed_in:.5f} in + ${routed_out:.5f} out = "
                f"**${routed_total:.5f}**  \n"
                f"**Baseline (`gpt-4o`):** ${baseline_in:.5f} in + ${baseline_out:.5f} out = "
                f"**${baseline_total:.5f}**  \n"
                f"**Savings on this prompt:** **{saved_pct:.1f}%**"
            )
            st.caption(
                "Costs are per 1k input/output tokens. Actual cost scales linearly with real token counts."
            )

        st.markdown("##### Extracted features the classifier used")
        feat_cols = st.columns(5)
        feat_cols[0].metric("Token count", features.token_count)
        feat_cols[1].metric("Instruction verbs", features.instruction_verb_count,
                            help="Words like 'analyze', 'compare', 'synthesize' — strong signal of complexity.")
        feat_cols[2].metric("Constraints", features.constraint_count,
                            help="Phrases like 'exactly N', 'in JSON', 'must'.")
        feat_cols[3].metric("Has context", "yes" if features.has_context else "no",
                            help="True if the prompt includes a long quoted block or > 200 words.")
        feat_cols[4].metric("Output complexity", features.output_format_complexity,
                            help="Count of structured-output hints (json, yaml, table, list).")


# =================== TAB 3: DRILL-DOWN ===================
with tab_drilldown:
    st.subheader("Recent requests")
    st.caption(
        "Browse and filter every logged request. "
        "Pick a row in the dropdown below to see the full prompt, verdict reasoning, and cost breakdown."
    )

    all_df = _all_requests(str(db_file))
    if all_df.empty:
        st.info("No requests yet.")
        st.stop()

    # Filters
    filter_cols = st.columns(3)
    with filter_cols[0]:
        tier_filter = st.multiselect(
            "Tier", options=sorted(all_df["tier"].unique()),
            default=sorted(all_df["tier"].unique()),
        )
    with filter_cols[1]:
        model_filter = st.multiselect(
            "Final model", options=sorted(all_df["final_model"].unique()),
            default=sorted(all_df["final_model"].unique()),
        )
    with filter_cols[2]:
        verdict_filter = st.multiselect(
            "Verdict", options=sorted(all_df["verdict"].unique()),
            default=sorted(all_df["verdict"].unique()),
        )

    filtered = all_df[
        all_df["tier"].isin(tier_filter)
        & all_df["final_model"].isin(model_filter)
        & all_df["verdict"].isin(verdict_filter)
    ]
    st.caption(f"{len(filtered)} of {len(all_df)} requests match the filters.")

    display = filtered[
        [
            "id", "timestamp", "tier", "candidate_model", "verdict",
            "escalated", "final_model", "final_cost", "baseline_cost",
            "prompt_preview",
        ]
    ]
    st.dataframe(display, width="stretch", height=320)

    # Per-request drill-down
    if not filtered.empty:
        ids = filtered["id"].tolist()
        chosen = st.selectbox("Inspect request id", options=ids, index=0)
        row = filtered[filtered["id"] == chosen].iloc[0]
        st.markdown("##### Request detail")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"**Tier:** `{row['tier']}`")
            st.markdown(f"**Candidate model:** `{row['candidate_model']}`")
            st.markdown(f"**Final model:** `{row['final_model']}`")
            st.markdown(f"**Escalated:** {'yes' if row['escalated'] else 'no'}")
        with d2:
            st.markdown(f"**Verdict:** `{row['verdict']}` "
                        f"(score={row['verdict_score']:.2f}, method=`{row['verdict_method']}`)")
            st.markdown(f"**Candidate cost:** ${row['candidate_cost']:.6f}")
            st.markdown(f"**Baseline cost:** ${row['baseline_cost']:.6f}")
            st.markdown(f"**Final cost:** ${row['final_cost']:.6f}")
        st.markdown("**Prompt preview:**")
        st.code(row["prompt_preview"], language="text")
