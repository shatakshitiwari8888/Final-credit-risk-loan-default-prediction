"""Client-facing Streamlit dashboard for loan portfolio analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from credit_risk.client_engine import (
    fit_borrower_model, format_period_range, load_portfolio,
    peer_benchmark, prepare_borrower_features, risk_band,
)

NAVIGATION = ["Portfolio Summary", "Risk & Segments", "Borrower Assessment"]
NAVY, NAVY_2, TEAL, GOLD, RED = "#14263D", "#1F3854", "#168C88", "#C59A52", "#B94A48"
INK, MUTED, GRID, BG = "#233247", "#718096", "#E7ECF1", "#F4F6F8"

st.set_page_config(page_title="Loan Portfolio Intelligence", page_icon="▦", layout="wide")
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
      .metric-foot {{ color: #98A3B0; font-size: .72rem; margin-top: .28rem; }}
      .result-band {{ background: #E6F4F2; border: 1px solid #C6E4E0; border-radius: 12px; padding: 17px 19px; }}
      .fine-print {{ font-size: .75rem; color: {MUTED}; line-height: 1.45; }}
      div[data-testid="stMetric"] {{ background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 12px 15px; }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 700; }}
      div[data-testid="stForm"] {{ background: white; border: 1px solid {GRID}; border-radius: 12px; padding: 18px 20px; }}
      .stFormSubmitButton > button[kind="primary"] {{ background: {NAVY}; border: 1px solid {NAVY}; border-radius: 8px; font-weight: 700; }}
      .stFormSubmitButton > button[kind="primary"]:hover {{ background: {TEAL}; border-color: {TEAL}; }}
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
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=22, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID),
    )
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
        '<div class="brand-mark">Credit portfolio services</div>'
        '<div class="brand-name">Loan Portfolio<br>Intelligence</div>'
        '<div class="brand-sub">Portfolio monitoring · Credit risk · Borrower review</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    selected_page = st.radio("Workspace", NAVIGATION, label_visibility="visible")
    st.divider()
    st.caption("PORTFOLIO COVERAGE")
    st.markdown(f"**{period_label}**")
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
        ("Funded exposure", compact_money(total_funded), "Total funded principal"),
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")

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
        st.metric("Funded exposure", compact_money(float(filtered[funded_column].fillna(0).sum())))
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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")

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
        st.plotly_chart(fig, width="stretch")
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
        st.plotly_chart(fig, width="stretch")

    st.caption("Segment rates are historical observations, not forecasts. Unresolved loans are excluded from charge-off-rate calculations.")


elif selected_page == "Borrower Assessment":
    header("Borrower Assessment", "Review an applicant against historical portfolio experience.", "Individual borrower review")
    st.markdown(
        '<div class="fine-print">Enter the applicant’s stated financial profile. Results are indicative and should be considered alongside verified documents, bureau records and the lender’s credit policy.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with st.form("borrower_assessment", border=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            annual_income = st.number_input("Gross annual income (USD)", min_value=1_000.0, max_value=5_000_000.0, value=75_000.0, step=2_500.0, format="%.0f")
            dti = st.number_input("Debt-to-income ratio (%)", min_value=0.0, max_value=100.0, value=18.0, step=0.5, format="%.1f")
        with c2:
            fico = st.slider("FICO score", min_value=300, max_value=850, value=700, step=5)
            loan_amount = st.number_input("Requested loan amount (USD)", min_value=500.0, max_value=500_000.0, value=15_000.0, step=500.0, format="%.0f")
        submit = st.form_submit_button("Run borrower assessment", type="primary", width="stretch")

    if submit:
        try:
            estimator = get_estimator()
            features = prepare_borrower_features(annual_income, dti, fico, loan_amount)
            probability = float(estimator.predict_proba(features)[0, 1])
            peers = peer_benchmark(portfolio, annual_income, dti, fico, loan_amount)
            st.session_state["client_assessment"] = {
                "probability": probability, "band": risk_band(probability), "peers": peers,
                "income": annual_income, "dti": dti, "fico": fico,
                "loan_amount": loan_amount, "lti": loan_amount / annual_income,
            }
        except Exception as exc:
            st.error(f"Unable to complete the assessment: {exc}")

    result = st.session_state.get("client_assessment")
    if result:
        st.write("")
        st.markdown("### Assessment result")
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Estimated probability of default", rate(result["probability"]))
        with b:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">Indicative risk band</div>'
                f'<div class="metric-value">{result["band"]}</div>'
                f'<div class="metric-foot">Based on the entered profile</div></div>',
                unsafe_allow_html=True,
            )
        with c:
            st.metric("Comparable historical loans", f'{result["peers"]["count"]:,}')
        with d:
            peer_rate = result["peers"]["default_rate"]
            st.metric("Comparable group charge-off rate", rate(peer_rate) if peer_rate is not None else "Not available")

        st.progress(float(np.clip(result["probability"], 0, 1)))
        st.markdown("#### Applicant profile")
        details = pd.DataFrame(
            [
                ("Annual income", money(result["income"])),
                ("Debt-to-income", f'{result["dti"]:.1f}%'),
                ("FICO score", f'{result["fico"]:,.0f}'),
                ("Requested principal", money(result["loan_amount"])),
                ("Loan-to-income", f'{result["lti"]:.2f}x'),
            ],
            columns=["Assessment input", "Applicant value"],
        )
        st.dataframe(details, hide_index=True, width="stretch")
        st.markdown(
            '<div class="result-band"><strong>Assessment context</strong><br>'
            'The estimated probability is a historical-data-based indicator. It is not an automated approval or decline, '
            'a quoted interest rate, or a substitute for a complete underwriting review.</div>',
            unsafe_allow_html=True,
        )


st.divider()
st.markdown(
    '<div class="fine-print">Historical charge-off rates are calculated only on loans with a resolved outcome. '
    'Borrower estimates are indicative decision-support information and are not a credit approval, decline or pricing offer.</div>',
    unsafe_allow_html=True,
)
