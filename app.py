"""Client-facing Streamlit dashboard for loan portfolio analytics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from credit_risk.client_engine import (
    fit_borrower_model, format_period_range, load_portfolio,
    peer_benchmark, prepare_borrower_features, risk_band,
    historical_grade_assessment, logistic_contributions,
    borrower_sensitivity, stress_test_sample, evaluate_borrower_model,
)

NAVIGATION = ["Portfolio Summary", "Risk & Segments", "Borrower Assessment", "Model Diagnostics"]
NAVY, NAVY_2, TEAL, GOLD, RED = "#14263D", "#1F3854", "#168C88", "#C59A52", "#B94A48"
INK, MUTED, GRID, BG = "#111111", "#111111", "#E7ECF1", "#F4F6F8"

st.set_page_config(page_title="Loan Portfolio Intelligence", page_icon=str(Path(__file__).parent / "assets" / "rupee.svg"), layout="wide")
st.markdown(
    f"""
    <style>
      .stApp {{ background: {BG}; color: {INK}; }}
      .block-container {{ max-width: 1500px; padding: 1.55rem 2.3rem 2.4rem; }}
      [data-testid="stSidebar"] {{ background: {NAVY}; border-right: 1px solid #203B57; }}
      [data-testid="stSidebar"] * {{ color: #F4F7FA !important; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding: .35rem .15rem; font-weight: 600; }}
      h1, h2, h3 {{ color: {NAVY}; letter-spacing: -.025em; }}
      h1 {{ font-size: 2rem !important; font-weight: 760 !important; }}
      h2 {{ font-size: 1.25rem !important; font-weight: 700 !important; }}
      .brand-mark {{ font-size: .68rem; font-weight: 800; letter-spacing: .18em; color: #91B5C5; text-transform: uppercase; }}
      .brand-name {{ color: #FFFFFF; font-size: 1.18rem; font-weight: 750; line-height: 1.2; margin-top: .28rem; }}
      .brand-sub {{ color: #B6C5D3; font-size: .77rem; margin-top: .38rem; }}
      .eyebrow {{ color: {TEAL}; text-transform: uppercase; letter-spacing: .14em; font-size: .68rem; font-weight: 800; margin-bottom: .35rem; }}
      .page-subtitle {{ color: {MUTED}; font-size: .93rem; margin-top: -.45rem; margin-bottom: 1.15rem; }}
      .metric-card {{ background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 15px 17px; min-height: 110px; box-shadow: 0 2px 8px rgba(15,35,55,.025); }}
      .metric-title {{ color: {MUTED}; font-size: .7rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }}
      .metric-value {{ color: {NAVY}; font-size: 1.55rem; font-weight: 780; margin-top: .38rem; line-height: 1.2; }}
      .metric-foot {{ color: #111111; font-size: .72rem; margin-top: .28rem; }}
      .result-band {{ background: #E6F4F2; border: 1px solid #C6E4E0; border-radius: 12px; padding: 17px 19px; }}
      .fine-print {{ font-size: .75rem; color: {MUTED}; line-height: 1.45; }}
      div[data-testid="stMetric"] {{ background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 12px 15px; }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 700; }}
      div[data-testid="stForm"] {{ background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 18px 20px; }}
      .stFormSubmitButton > button[kind="primary"] {{ background: {NAVY}; border: 1px solid {NAVY}; border-radius: 8px; font-weight: 700; }}
      .stFormSubmitButton > button[kind="primary"]:hover {{ background: {TEAL}; border-color: {TEAL}; }}
      /* High-contrast main-page typography; preserve sidebar and dark inputs. */
      [data-testid="stMain"] p,
      [data-testid="stMain"] label,
      [data-testid="stMain"] [data-testid="stCaptionContainer"],
      [data-testid="stMain"] [data-testid="stCaptionContainer"] *,
      [data-testid="stMain"] [data-testid="stMarkdownContainer"],
      [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stMain"] [data-testid="stMetricLabel"],
      [data-testid="stMain"] [data-testid="stMetricLabel"] *,
      [data-testid="stMain"] [data-testid="stMetricValue"],
      [data-testid="stMain"] [data-testid="stMetricValue"] *,
      [data-testid="stMain"] .metric-title,
      [data-testid="stMain"] .metric-value,
      [data-testid="stMain"] .metric-foot,
      [data-testid="stMain"] .page-subtitle,
      [data-testid="stMain"] .fine-print {{
        color: #111111 !important;
      }}
      /* Keep text readable on dark input controls and the coloured button. */
      [data-testid="stMain"] [data-baseweb="input"] input,
      [data-testid="stMain"] [data-baseweb="select"] input,
      [data-testid="stMain"] .stFormSubmitButton button,
      [data-testid="stMain"] .stFormSubmitButton button * {{
        color: #FFFFFF !important;
      }}
      /* Explicit metric label and value overrides for current Streamlit. */
      [data-testid="stMain"] [data-testid="stMetric"] {{
        color: #111111 !important;
      }}
      /* Allow selected filter chips to wrap inside the three multiselects. */
      [data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
        height: auto !important;
        min-height: 42px !important;
        max-height: none !important;
        flex-wrap: wrap !important;
        align-items: flex-start !important;
        overflow: visible !important;
      }}
      [data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div > div {{
        flex-wrap: wrap !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
        white-space: normal !important;
      }}
      [data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {{
        max-width: 100%;
        white-space: normal !important;
        height: auto !important;
        min-height: 26px;
      }}
      [data-testid="stMain"] [data-testid="stMultiSelect"] [data-baseweb="tag"] span {{
        white-space: normal !important;
        overflow-wrap: anywhere !important;
      }}
      hr {{ border-color: {GRID}; }}
      footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Preparing portfolio information…")
def get_portfolio() -> pd.DataFrame:
    return load_portfolio()


@st.cache_resource(show_spinner="Preparing borrower assessment…")
def get_estimator():
    return fit_borrower_model(get_portfolio())


def money(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return "$" + format(value, f",.{decimals}f")


def compact_money(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1_000_000_000:
        return "$" + f"{value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return "$" + f"{value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return "$" + f"{value / 1_000:,.0f}K"
    return money(value)


def rate(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def chart_layout(fig, height=340):
    """Enforce black Plotly text regardless of browser light/dark theme."""
    fig.update_layout(
        template="plotly_white",
        height=height, margin=dict(l=8, r=12, t=25, b=16),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#111111", size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color="#111111", size=13),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, linecolor=GRID,
            tickfont=dict(color="#111111", size=12),
            title_font=dict(color="#111111", size=13),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID,
            tickfont=dict(color="#111111", size=12),
            title_font=dict(color="#111111", size=13),
            automargin=True,
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#111111"),
            title_font=dict(color="#111111"),
        ),
    )
    fig.update_traces(selector=dict(type="pie"), textfont=dict(color="#FFFFFF", size=13))
    return fig


def header(title: str, subtitle: str, eyebrow: str):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(title: str, value: str, foot: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">{title}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>',
        unsafe_allow_html=True,
    )


def resolved_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["target"].notna()]


def purpose_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


try:
    portfolio = get_portfolio()
except Exception:
    st.error("Portfolio information is not available.")
    st.caption("The dashboard is ready for the LendingClub accepted-loan file. The data connection must be completed before client access.")
    st.stop()

resolved = resolved_only(portfolio)
default_rate = float(resolved["target"].mean()) if len(resolved) else np.nan
funded_column = "funded_amnt" if "funded_amnt" in portfolio.columns else "loan_amnt"
total_funded = float(portfolio[funded_column].fillna(0).sum())
period_label = format_period_range(portfolio)

with st.sidebar:
    st.markdown(
        '<div class="brand-mark">₹ · Credit portfolio services</div>'
        '<div class="brand-name">Loan Portfolio<br>Intelligence</div>'
        '<div class="brand-sub">Portfolio monitoring · Credit risk · Borrower review</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    selected_page = st.radio("Workspace", NAVIGATION, label_visibility="visible")
    st.divider()
    st.caption("PORTFOLIO COVERAGE")
    st.markdown(f"**{period_label}**")
    if portfolio.attrs.get("sampled"):
        st.caption(f"{len(portfolio):,} sampled records from {portfolio.attrs.get('source_rows', 0):,} source loans")
        st.warning("Large dataset: charts and totals reflect a reproducible 4% sample, not the full population.", icon="ℹ️")
    else:
        st.caption(f"{len(portfolio):,} accepted loan records")
    st.markdown('<div class="fine-print">Historical outcomes use loans with a resolved terminal status.</div>', unsafe_allow_html=True)


if selected_page == "Portfolio Summary":
    header("Portfolio Summary", "A consolidated view of loan exposure, repayment outcomes and portfolio quality.", "Portfolio intelligence")
    total_loans = len(portfolio)
    average_loan = float(portfolio["loan_amnt"].mean())
    median_fico = float(portfolio["fico_range_low"].median())
    cols = st.columns(5)
    values = [
        ("Loans analysed", f"{total_loans:,}", f"Accepted loans · {period_label}"),
        ("Sample funded exposure" if portfolio.attrs.get("sampled") else "Funded exposure", compact_money(total_funded), "Sum of analysed records"),
        ("Observed charge-off rate", rate(default_rate), f"Resolved loans · {len(resolved):,}"),
        ("Average loan size", money(average_loan), "Original loan principal"),
        ("Median FICO", f"{median_fico:,.0f}" if pd.notna(median_fico) else "—", "At origination"),
    ]
    for col, item in zip(cols, values):
        with col:
            metric_card(*item)

    st.write("")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown("### Repayment outcomes")
        outcome_counts = (
            resolved["outcome"].value_counts()
            .reindex(["Fully paid", "Charged off"])
            .fillna(0)
            .rename_axis("Outcome")
            .reset_index(name="Loans")
        )
        fig = px.pie(
            outcome_counts, names="Outcome", values="Loans", hole=0.66,
            color="Outcome", color_discrete_map={"Fully paid": TEAL, "Charged off": RED},
        )
        fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="white", width=2)))
        chart_layout(fig, 345)
        st.plotly_chart(fig, width="stretch", theme=None)

    with right:
        st.markdown("### Charge-off rate by origination year")
        vintage = (
            resolved.dropna(subset=["issue_year"])
            .groupby("issue_year", observed=True)
            .agg(Loans=("target", "size"), Charge_off_rate=("target", "mean"))
            .reset_index()
            .sort_values("issue_year")
        )
        vintage["issue_year"] = vintage["issue_year"].astype(int).astype(str)
        fig = px.line(
            vintage, x="issue_year", y="Charge_off_rate", markers=True,
            labels={"issue_year": "Origination year", "Charge_off_rate": "Observed charge-off rate"},
        )
        fig.update_traces(line=dict(color=TEAL, width=3), marker=dict(size=7, color=GOLD))
        fig.update_yaxes(tickformat=".0%")
        chart_layout(fig, 345)
        st.plotly_chart(fig, width="stretch", theme=None)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Portfolio by credit grade")
        grade_summary = (
            portfolio.assign(grade=portfolio["grade"].astype("string"))
            .groupby("grade", observed=True)
            .agg(Loans=("loan_amnt", "size"), Funded=("funded_amnt", "sum"))
            .reindex(list("ABCDEFG"))
            .dropna(how="all")
            .reset_index()
        )
        fig = px.bar(
            grade_summary, x="grade", y="Loans",
            labels={"grade": "Origination grade", "Loans": "Loan count"},
            color_discrete_sequence=[NAVY_2],
        )
        fig.update_traces(marker_line_width=0)
        chart_layout(fig, 310)
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.markdown("### Lending purpose")
        purpose_summary = (
            portfolio.assign(Purpose=portfolio["purpose"].astype("string").map(purpose_label))
            .groupby("Purpose", observed=True)
            .agg(Loans=("loan_amnt", "size"), Funded=("funded_amnt", "sum"))
            .sort_values("Loans", ascending=False)
            .head(8)
            .reset_index()
            .sort_values("Loans")
        )
        fig = px.bar(
            purpose_summary, x="Loans", y="Purpose", orientation="h",
            labels={"Loans": "Loan count"}, color_discrete_sequence=[TEAL],
        )
        fig.update_traces(marker_line_width=0)
        chart_layout(fig, 310)
        st.plotly_chart(fig, width="stretch", theme=None)

    st.caption(
        "Observed charge-off rate is calculated from loans with a terminal outcome (Fully Paid or Charged Off). "
        "Current and other unresolved accounts are excluded from this rate."
    )


elif selected_page == "Risk & Segments":
    header("Risk & Segments", "Compare credit quality and observed repayment outcomes across portfolio segments.", "Portfolio analysis")
    min_year = int(portfolio["issue_year"].dropna().min())
    max_year = int(portfolio["issue_year"].dropna().max())
    f1, f2, f3, f4 = st.columns([1.2, 1.1, 1.5, 1.4])
    with f1:
        years = st.slider("Origination years", min_year, max_year, (min_year, max_year))
    with f2:
        grades = sorted(portfolio["grade"].astype("string").dropna().unique().tolist())
        selected_grades = st.multiselect("Credit grade", grades, default=grades)
    with f3:
        purposes = sorted(portfolio["purpose"].astype("string").dropna().unique().tolist())
        selected_purposes = st.multiselect(
            "Loan purpose", purposes,
            default=[x for x in purposes if x in {"debt_consolidation", "credit_card", "home_improvement", "major_purchase"}] or purposes[:4],
            format_func=purpose_label,
        )
    with f4:
        states = sorted(portfolio["addr_state"].astype("string").dropna().unique().tolist())
        selected_states = st.multiselect("Borrower state", states, default=states)

    filtered = portfolio.loc[
        portfolio["issue_year"].between(years[0], years[1])
        & portfolio["grade"].astype("string").isin(selected_grades)
        & portfolio["purpose"].astype("string").isin(selected_purposes)
        & portfolio["addr_state"].astype("string").isin(selected_states)
    ]
    filtered_resolved = resolved_only(filtered)
    filtered_rate = float(filtered_resolved["target"].mean()) if len(filtered_resolved) else np.nan
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Loans in selection", f"{len(filtered):,}")
    with m2:
        st.metric("Sample funded exposure" if portfolio.attrs.get("sampled") else "Funded exposure", compact_money(float(filtered[funded_column].fillna(0).sum())))
    with m3:
        st.metric("Observed charge-off rate", rate(filtered_rate))
    with m4:
        st.metric("Median FICO", f"{filtered['fico_range_low'].median():,.0f}" if filtered["fico_range_low"].notna().any() else "—")

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Charge-off rate by origination grade")
        grade_stats = (
            filtered_resolved.assign(grade=filtered_resolved["grade"].astype("string"))
            .groupby("grade", observed=True)
            .agg(Loans=("target", "size"), Charge_off_rate=("target", "mean"))
            .reindex(list("ABCDEFG"))
            .dropna(subset=["Loans"])
            .reset_index()
        )
        fig = px.bar(
            grade_stats, x="grade", y="Charge_off_rate",
            labels={"grade": "Origination grade", "Charge_off_rate": "Observed charge-off rate"},
            color="Charge_off_rate", color_continuous_scale=["#8FC9BF", TEAL, NAVY],
        )
        fig.update_traces(text=grade_stats["Charge_off_rate"].map(lambda x: f"{x:.1%}"), textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(coloraxis_showscale=False)
        chart_layout(fig, 350)
        st.plotly_chart(fig, width="stretch", theme=None)

    with right:
        st.markdown("### Charge-off rate by debt-to-income")
        debt = filtered_resolved.copy()
        debt["DTI band"] = pd.cut(
            debt["dti"], bins=[-0.01, 10, 20, 30, 40, 50, np.inf],
            labels=["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", "50%+"],
        )
        dti_stats = (
            debt.groupby("DTI band", observed=True)
            .agg(Loans=("target", "size"), Charge_off_rate=("target", "mean"))
            .reset_index()
        )
        fig = px.bar(
            dti_stats, x="DTI band", y="Charge_off_rate",
            labels={"DTI band": "Debt-to-income band", "Charge_off_rate": "Observed charge-off rate"},
            color_discrete_sequence=[TEAL],
        )
        fig.update_traces(text=dti_stats["Charge_off_rate"].map(lambda x: f"{x:.1%}"), textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        chart_layout(fig, 350)
        st.plotly_chart(fig, width="stretch", theme=None)

    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Charge-off rate by FICO band")
        fico = filtered_resolved.copy()
        fico["FICO band"] = pd.cut(
            fico["fico_range_low"], bins=[0, 620, 660, 700, 740, 780, 900],
            labels=["Below 620", "620–659", "660–699", "700–739", "740–779", "780+"],
        )
        fico_stats = (
            fico.groupby("FICO band", observed=True)
            .agg(Loans=("target", "size"), Charge_off_rate=("target", "mean"))
            .reset_index()
        )
        fig = px.bar(
            fico_stats, x="FICO band", y="Charge_off_rate",
            labels={"FICO band": "FICO band", "Charge_off_rate": "Observed charge-off rate"},
            color_discrete_sequence=[NAVY_2],
        )
        fig.update_traces(text=fico_stats["Charge_off_rate"].map(lambda x: f"{x:.1%}"), textposition="outside")
        fig.update_yaxes(tickformat=".0%")
        chart_layout(fig, 330)
        st.plotly_chart(fig, width="stretch", theme=None)
    with right:
        st.markdown("### Loan term mix")
        term_stats = (
            filtered.assign(Term=filtered["term_months"].map({36: "36 months", 60: "60 months"}).fillna("Other"))
            .groupby("Term", observed=True)
            .agg(Loans=("loan_amnt", "size"), Funded=("funded_amnt", "sum"))
            .reset_index()
        )
        fig = px.pie(
            term_stats, names="Term", values="Loans", hole=0.58,
            color="Term", color_discrete_map={"36 months": TEAL, "60 months": NAVY_2, "Other": GOLD},
        )
        fig.update_traces(textinfo="percent+label")
        chart_layout(fig, 330)
        st.plotly_chart(fig, width="stretch", theme=None)


    st.markdown("### FICO × debt-to-income risk heatmap")
    st.caption("Observed charge-off rates among resolved historical loans in the selected sample. Blank cells have fewer than 15 resolved loans.")
    matrix = filtered_resolved[["fico_range_low", "dti", "target"]].copy()
    matrix["FICO tier"] = pd.cut(
        matrix["fico_range_low"], [0, 650, 700, 750, 900],
        labels=["Below 650", "650–699", "700–749", "750+"], right=False,
    )
    matrix["DTI tier"] = pd.cut(
        matrix["dti"], [-0.001, 15, 20, 25, np.inf],
        labels=["Below 15%", "15–20%", "20–25%", "25%+"], right=False,
    )
    cohorts = (
        matrix.groupby(["FICO tier", "DTI tier"], observed=True)["target"]
        .agg(default_rate="mean", loans="size").reset_index()
    )
    cohorts.loc[cohorts["loans"] < 15, "default_rate"] = np.nan
    display = cohorts.pivot(index="FICO tier", columns="DTI tier", values="default_rate")
    fig = px.imshow(
        display.astype(float), text_auto=".0%", color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=0.50, aspect="auto", labels={"color": "Default rate"},
    )
    fig.update_layout(margin=dict(l=5, r=5, t=10, b=5), height=320)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("Macroeconomic scenario simulator (illustrative)", expanded=False):
        st.caption(
            "Model-based sensitivity on at most 5,000 sampled loans, not a "
            "regulatory stress test or full-portfolio forecast. Loss given default "
            "(LGD) is an assumption; no recovery data is modeled."
        )
        x1, x2, x3 = st.columns(3)
        with x1:
            shock_income = st.slider("Borrower income reduction (%)", 0, 40, 10, 5)
        with x2:
            shock_dti = st.slider("DTI increase (percentage points)", 0, 25, 5, 1)
        with x3:
            assumed_lgd = st.slider("Assumed loss given default (%)", 20, 90, 45, 5)
        if st.button("Calculate scenario impact", key="run_portfolio_stress"):
            try:
                with st.spinner("Scoring sample under both scenarios…"):
                    stress = stress_test_sample(
                        portfolio, get_estimator(), shock_income, shock_dti,
                        assumed_lgd / 100,
                    )
                st.session_state["portfolio_stress"] = {
                    "inputs": (shock_income, shock_dti, assumed_lgd),
                    "result": stress,
                }
            except Exception as exc:
                st.error(f"Scenario unavailable: {exc}")
        scenario = st.session_state.get("portfolio_stress")
        if scenario:
            if scenario["inputs"] != (shock_income, shock_dti, assumed_lgd):
                st.info("Scenario inputs changed. Click Calculate to update the displayed results.")
            stress = scenario["result"]
            st.caption(f"Scenario basis: {stress['loans']:,} sampled loans · {compact_money(stress['exposure'])} sample principal.")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Baseline modeled PD", rate(stress["baseline_pd"]))
            with q2:
                st.metric("Scenario modeled PD", rate(stress["stressed_pd"]),
                          delta=f"{(stress['stressed_pd']-stress['baseline_pd'])*100:+.2f} pp",
                          delta_color="inverse")
            with q3:
                st.metric("Incremental sample expected loss", money(stress["incremental_el"]))
            st.caption(
                f"Baseline sample expected loss: {money(stress['baseline_el'])} · "
                f"Scenario expected loss: {money(stress['stressed_el'])}. "
                "Calculated from modeled PD × sampled principal × assumed LGD."
            )
    st.caption("Segment rates are historical observations, not forecasts. Unresolved loans are excluded from charge-off-rate calculations.")


elif selected_page == "Borrower Assessment":
    header(
        "Retail Loan Applicant Risk Scorer",
        "Historical borrower risk, grade comparison and transparent model explanation.",
        "Borrower risk assessment & underwriting",
    )
    st.markdown(
        '<div class="fine-print">The grade below is an indicative alignment with LendingClub historical origination grades, '
        'not an official bureau or lender-assigned grade, approval, decline, or interest-rate offer.</div>',
        unsafe_allow_html=True,
    )

    profiles = {
        "Stronger profile": (110_000.0, 9.0, 780, 12_000.0),
        "Typical profile": (75_000.0, 18.0, 700, 15_000.0),
        "Higher-risk profile": (35_000.0, 38.0, 600, 30_000.0),
    }

    def apply_profile(profile_name: str):
        income, debt, score, principal = profiles[profile_name]
        st.session_state["assess_income"] = income
        st.session_state["assess_dti"] = debt
        st.session_state["assess_fico"] = score
        st.session_state["assess_principal"] = principal
        st.session_state.pop("client_assessment", None)

    for key, default in [
        ("assess_income", 75_000.0), ("assess_dti", 18.0),
        ("assess_fico", 700), ("assess_principal", 15_000.0),
    ]:
        st.session_state.setdefault(key, default)

    st.markdown("#### Example borrower profiles")
    pcols = st.columns(3)
    for column, profile_name in zip(pcols, profiles):
        with column:
            st.button(profile_name, on_click=apply_profile,
                      args=(profile_name,), use_container_width=True)

    with st.form("borrower_assessment", border=True):
        st.markdown("#### Applicant financial variables")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            annual_income = st.number_input(
                "Gross annual income (USD)", min_value=1_000.0,
                max_value=5_000_000.0, step=2_500.0, format="%.0f",
                key="assess_income",
            )
            dti = st.slider(
                "Debt-to-income ratio (%)", min_value=0.0, max_value=100.0,
                step=0.5, key="assess_dti",
            )
        with c2:
            fico = st.slider(
                "FICO credit score", min_value=300, max_value=850, step=5,
                key="assess_fico",
            )
            loan_amount = st.number_input(
                "Requested loan amount (USD)", min_value=500.0,
                max_value=500_000.0, step=500.0, format="%.0f",
                key="assess_principal",
            )
        live_lti = loan_amount / max(annual_income, 1.0)
        st.caption(f"Engineered feature · Loan-to-income ratio: {live_lti:.3f}x")
        submit = st.form_submit_button(
            "Run borrower assessment", type="primary", use_container_width=True
        )

    if submit:
        try:
            estimator = get_estimator()
            features = prepare_borrower_features(
                annual_income, dti, fico, loan_amount
            )
            probability = float(estimator.predict_proba(features)[0, 1])
            peers = peer_benchmark(
                portfolio, annual_income, dti, fico, loan_amount
            )
            grade_info = historical_grade_assessment(portfolio, probability)
            explanation = logistic_contributions(estimator, features)
            st.session_state["client_assessment"] = {
                "probability": probability,
                "band": risk_band(probability),
                "peers": peers,
                "grade": grade_info["grade"],
                "reference": grade_info["reference"],
                "explanation": explanation,
                "income": annual_income, "dti": dti, "fico": fico,
                "loan_amount": loan_amount, "lti": live_lti,
            }
        except Exception as exc:
            st.session_state.pop("client_assessment", None)
            st.error(f"Unable to complete the assessment: {exc}")

    result = st.session_state.get("client_assessment")
    if result:
        st.write("")
        st.markdown("### Borrower risk assessment")
        a, b, c = st.columns(3)
        with a:
            st.metric("Indicative historical grade", result["grade"])
        with b:
            st.metric("Estimated probability of default", rate(result["probability"]))
        with c:
            st.metric("Risk category", result["band"])

        p1, p2 = st.columns(2)
        with p1:
            st.metric("Comparable historical loans", f'{result["peers"]["count"]:,}')
        with p2:
            peer_rate = result["peers"]["default_rate"]
            st.metric(
                "Comparable group charge-off rate",
                rate(peer_rate) if peer_rate is not None else "Not available",
            )
        st.progress(float(np.clip(result["probability"], 0, 1)))
        st.caption(
            "Grade A–G is aligned to the sample's observed historical charge-off "
            "rates by LendingClub origination grade. It is not an assigned lender "
            "grade or a credit decision."
        )

        st.markdown("#### Applicant financial profile")
        details = pd.DataFrame(
            [
                ("Annual income", money(result["income"])),
                ("Debt-to-income", f'{result["dti"]:.1f}%'),
                ("FICO score", f'{result["fico"]:,.0f}'),
                ("Requested principal", money(result["loan_amount"])),
                ("Loan-to-income", f'{result["lti"]:.3f}x'),
            ],
            columns=["Assessment input", "Applicant value"],
        )
        st.dataframe(details, hide_index=True, use_container_width=True)

        st.markdown("#### What-if borrower improvement simulator")
        st.caption("Model sensitivity only: changing inputs does not guarantee a lower rate, credit approval, or a grade assigned by a lender.")
        w1, w2, w3 = st.columns(3)
        with w1:
            extra_income = st.slider("Income increase ($)", 0, 30000, 5000, 1000, key="whatif_income")
        with w2:
            dti_drop = st.slider("DTI reduction (points)", 0.0, 15.0, 3.0, 0.5, key="whatif_dti")
        with w3:
            fico_boost = st.slider("FICO improvement (points)", 0, 80, 20, 5, key="whatif_fico")
        try:
            sensitivity = borrower_sensitivity(
                get_estimator(), result["income"], result["dti"],
                result["fico"], result["loan_amount"],
                extra_income, dti_drop, fico_boost,
            )
            simulated_grade = historical_grade_assessment(
                portfolio, sensitivity["simulated_pd"]
            )["grade"]
            w4, w5, w6 = st.columns(3)
            with w4:
                st.metric("Current modeled PD", rate(sensitivity["baseline_pd"]))
            with w5:
                st.metric("What-if modeled PD", rate(sensitivity["simulated_pd"]),
                          delta=f"{(sensitivity['simulated_pd']-sensitivity['baseline_pd'])*100:+.2f} pp",
                          delta_color="inverse")
            with w6:
                st.metric("Indicative grade shift", f'{result["grade"]} → {simulated_grade}')
        except Exception as exc:
            st.error(f"What-if assessment unavailable: {exc}")

        memo = (
            "LOAN PORTFOLIO INTELLIGENCE — BORROWER RISK SUMMARY\\n"
            "Indicative research analysis; NOT a credit approval, loan offer or APR.\\n"
            f"Income: {money(result['income'])} | DTI: {result['dti']:.1f}% | "
            f"FICO: {result['fico']:.0f} | Requested principal: {money(result['loan_amount'])}\\n"
            f"Estimated default probability: {result['probability']:.2%}\\n"
            f"Indicative historical grade: {result['grade']}\\n"
            f"Historical comparable loans: {result['peers']['count']:,}\\n"
            "Method: Logistic regression on resolved LendingClub loan data, with historical-grade comparison.\\n"
        )
        st.download_button(
            "Download assessment memo (.txt)", data=memo.encode("utf-8"),
            file_name="borrower_risk_assessment.txt", mime="text/plain",
            key="download_underwriting_memo",
        )

        st.markdown("#### Sigmoid risk curve")
        st.caption(
            "The red point represents the model's calculated log-odds and "
            "probability of default for this applicant."
        )
        z = np.linspace(-5.0, 5.0, 201)
        curve = pd.DataFrame({
            "Log-odds (z)": z,
            "Probability of default": 1 / (1 + np.exp(-z)),
        })
        fig = px.line(
            curve, x="Log-odds (z)", y="Probability of default",
            color_discrete_sequence=[NAVY],
        )
        fig.add_scatter(
            x=[result["explanation"]["log_odds"]],
            y=[result["probability"]],
            mode="markers+text",
            text=[f'{result["probability"]:.1%} PD'],
            textposition="top center",
            marker=dict(color=RED, size=12),
            name="Applicant",
        )
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        chart_layout(fig, 330)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Model explanation: standardized inputs to log-odds")
        st.caption(
            "These contributions are calculated from this app's fitted logistic "
            "regression coefficients, not from example or illustrative weights. "
            "Negative contributions reduce estimated log-odds of default."
        )
        decomposition = result["explanation"]["rows"].copy()
        decomposition["Input"] = decomposition["Input"].map(lambda v: f"{v:,.3f}")
        for column in ["Z-score", "Coefficient", "Log-odds contribution"]:
            decomposition[column] = decomposition[column].map(lambda v: f"{v:+.3f}")
        st.dataframe(decomposition, hide_index=True, use_container_width=True)
        intercept = result["explanation"]["intercept"]
        log_odds = result["explanation"]["log_odds"]
        st.caption(
            f"Intercept: {intercept:+.3f} · Total log-odds: {log_odds:+.3f} · "
            f"Predicted default probability: {result['probability']:.2%}"
        )

        reference = result["reference"]
        if not reference.empty:
            with st.expander("How is the indicative grade derived?"):
                st.write(
                    "The model estimates default probability from income, DTI, "
                    "FICO, loan amount and loan-to-income. That probability is "
                    "compared against historical charge-off rates for actual "
                    "LendingClub grades A–G. The closest ordered risk interval "
                    "determines the indicative historical grade; a lender may "
                    "assign a different grade."
                )
                summary = reference.reset_index().rename(columns={
                    "grade": "Historical grade",
                    "loans": "Resolved loans",
                    "observed_rate": "Historical charge-off rate",
                })
                st.dataframe(
                    summary, hide_index=True, use_container_width=True,
                    column_config={
                        "Historical charge-off rate": st.column_config.NumberColumn(
                            format="%.1%%"
                        )
                    },
                )

        st.markdown(
            '<div class="result-band"><strong>Assessment context</strong>'
            '<p>This is an indicative historical-data-based analysis. It is not '
            'a credit approval, decline, offered APR or financial advice. '
            'Review against verified borrower documents and lender policy.</p></div>',
            unsafe_allow_html=True,
        )


elif selected_page == "Model Diagnostics":
    header(
        "Model Performance & Validation",
        "Out-of-sample evaluation of the same logistic-regression model family.",
        "Responsible model validation",
    )
    st.caption(
        "Metrics are calculated on an untouched 20% holdout of up to 15,000 "
        "resolved historical loans. This diagnostic model is separately "
        "trained; numbers are not copied from the reference website."
    )

    @st.cache_resource(show_spinner="Running holdout model validation…")
    def get_holdout_metrics():
        return evaluate_borrower_model(get_portfolio())

    if st.button("Run holdout model validation", type="primary"):
        try:
            st.session_state["show_diagnostics"] = True
            get_holdout_metrics()
        except Exception as exc:
            st.session_state["show_diagnostics"] = False
            st.error(f"Model validation unavailable: {exc}")
    if st.session_state.get("show_diagnostics"):
        diagnostics = get_holdout_metrics()
        st.caption(f"Training records: {diagnostics['train_rows']:,} · Independent holdout: {diagnostics['test_rows']:,} · Classification threshold: {diagnostics['threshold']:.0%}.")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("ROC AUC", f'{diagnostics["auc"]:.3f}')
            st.metric("Holdout accuracy", rate(diagnostics["accuracy"]))
        with k2:
            st.metric("Default recall", rate(diagnostics["recall"]))
            st.metric("Default precision", rate(diagnostics["precision"]))
        with k3:
            st.metric("F1 score", f'{diagnostics["f1"]:.3f}')
            st.metric("Brier score (lower better)", f'{diagnostics["brier"]:.3f}')
        left, right = st.columns(2)
        with left:
            st.markdown("#### Holdout ROC curve")
            roc = pd.DataFrame({
                "False positive rate": diagnostics["fpr"],
                "True positive rate": diagnostics["tpr"],
            })
            fig = px.line(roc, x="False positive rate", y="True positive rate",
                          color_discrete_sequence=[TEAL])
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="#AAAAAA", dash="dash"))
            chart_layout(fig, 320)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        with right:
            st.markdown("#### Holdout confusion matrix")
            confusion = pd.DataFrame([
                {"Actual": "Repaid", "Predicted repaid": diagnostics["tn"],
                 "Predicted default": diagnostics["fp"]},
                {"Actual": "Defaulted", "Predicted repaid": diagnostics["fn"],
                 "Predicted default": diagnostics["tp"]},
            ])
            st.dataframe(confusion, hide_index=True, use_container_width=True)
            st.caption("False negatives are defaulted loans classified below the chosen PD threshold. Changing the threshold changes these counts.")
        st.markdown("#### Probability calibration")
        fig = px.scatter(
            diagnostics["calibration"], x="predicted_pd", y="observed_default",
            size="loans", labels={"predicted_pd": "Mean predicted PD",
                                  "observed_default": "Observed default rate"},
            color_discrete_sequence=[TEAL],
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="#999999", dash="dash"))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        chart_layout(fig, 360)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption("Performance is based on historical resolved loans. No lender approval or profitability claims are implied.")



st.divider()
st.markdown(
    '<div class="fine-print">Historical charge-off rates are calculated only on loans with a resolved outcome. '
    'Borrower estimates are indicative decision-support information and are not a credit approval, decline or pricing offer.</div>',
    unsafe_allow_html=True,
)
