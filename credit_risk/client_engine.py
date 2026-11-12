"""Client-facing LendingClub portfolio data and borrower assessment utilities.

This module intentionally exposes business outputs only. Model fitting and
preprocessing remain behind the dashboard interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "accepted_2007_to_2018Q4.csv.gz"

RAW_FIELDS = [
    "loan_status", "loan_amnt", "funded_amnt", "term", "int_rate",
    "installment", "grade", "sub_grade", "emp_length", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "purpose", "dti",
    "fico_range_low", "fico_range_high", "addr_state",
]

DISPLAY_FIELDS = [
    "loan_status", "loan_amnt", "funded_amnt", "term", "int_rate",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "purpose", "dti",
    "fico_range_low", "fico_range_high", "addr_state",
]

FEATURES = ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti"]
MAX_TRAINING_ROWS = 250_000


def _numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float32")


def _parse_month(value: pd.Series) -> pd.Series:
    text = value.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%b-%Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def load_portfolio(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read the accepted-loan archive, retaining only fields needed for client analytics."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "The LendingClub portfolio file is not available. "
            "Add data/accepted_2007_to_2018Q4.csv.gz using Git LFS."
        )

    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    available = [column for column in DISPLAY_FIELDS if column in header]
    required = {"loan_status", "loan_amnt", "annual_inc", "dti", "fico_range_low"}
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(
            "The uploaded file is missing required portfolio fields: " + ", ".join(missing)
        )

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=available,
        chunksize=200_000,
        low_memory=False,
        dtype_backend="pyarrow",
    ):
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame(columns=available)

    # Keep only loan-level variables suitable for portfolio reporting.
    defaults = {
        "funded_amnt": "loan_amnt",
        "term": None,
        "int_rate": None,
        "grade": "Not available",
        "sub_grade": "Not available",
        "emp_length": "Not available",
        "home_ownership": "Not available",
        "verification_status": "Not available",
        "issue_d": None,
        "purpose": "Not available",
        "fico_range_high": np.nan,
        "addr_state": "Not available",
    }
    for column, fallback in defaults.items():
        if column not in frame.columns:
            frame[column] = frame["loan_amnt"] if fallback == "loan_amnt" else fallback

    text_fields = [
        "loan_status", "term", "grade", "sub_grade", "emp_length",
        "home_ownership", "verification_status", "issue_d", "purpose", "addr_state",
    ]
    for column in text_fields:
        frame[column] = frame[column].astype("string").str.strip()

    numeric_fields = [
        "loan_amnt", "funded_amnt", "annual_inc", "dti",
        "fico_range_low", "fico_range_high", "int_rate", "installment",
    ]
    for column in numeric_fields:
        if column in frame.columns:
            frame[column] = _numeric(frame[column])

    if "term" in frame.columns:
        frame["term_months"] = pd.to_numeric(
            frame["term"].astype("string").str.extract(r"(\d+)")[0],
            errors="coerce",
        ).fillna(0).astype("int16")
    else:
        frame["term_months"] = np.int16(0)

    status = frame["loan_status"].fillna("").str.lower()
    is_default = status.str.contains(r"charged off|\bdefault\b", regex=True, na=False)
    is_paid = status.str.contains("fully paid", regex=False, na=False)
    resolved = is_default | is_paid

    frame["outcome"] = np.select(
        [is_default, is_paid],
        ["Charged off", "Fully paid"],
        default="Unresolved",
    )
    frame["target"] = np.where(resolved, is_default.astype("float32"), np.nan).astype("float32")
    frame["issue_date"] = _parse_month(frame["issue_d"]) if "issue_d" in frame.columns else pd.NaT
    frame["issue_year"] = frame["issue_date"].dt.year.astype("Int16")
    frame["issue_month"] = frame["issue_date"].dt.to_period("M").astype("string")
    frame["lti"] = (
        frame["loan_amnt"] / frame["annual_inc"].where(frame["annual_inc"] > 0)
    ).replace([np.inf, -np.inf], np.nan).astype("float32")

    # Normalize categorical labels once, keeping the full 2.26M-row archive compact.
    for column in [
        "outcome", "grade", "sub_grade", "purpose", "home_ownership",
        "emp_length", "verification_status", "addr_state",
    ]:
        frame[column] = frame[column].fillna("Not available").astype("category")

    # The raw loan_status string is not required in the client interface.
    frame.drop(columns=["loan_status", "term", "issue_d"], inplace=True, errors="ignore")
    frame["term_months"] = frame["term_months"].astype("int16")
    return frame


def prepare_borrower_features(
    annual_income: float,
    dti: float,
    fico: float,
    loan_amount: float,
) -> pd.DataFrame:
    income = max(float(annual_income), 1_000.0)
    principal = max(float(loan_amount), 500.0)
    debt_ratio = float(np.clip(dti, 0.0, 100.0))
    bureau_score = float(np.clip(fico, 300.0, 850.0))
    lti = float(np.clip(principal / income, 0.0, 5.0))
    return pd.DataFrame(
        [[income, debt_ratio, bureau_score, principal, lti]],
        columns=FEATURES,
    )


def fit_borrower_model(portfolio: pd.DataFrame):
    """Fit an indicative PD model on a reproducible sample of resolved historical loans."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    eligible = portfolio.loc[
        portfolio["target"].notna(),
        ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti", "target"],
    ].copy()
    eligible = eligible.replace([np.inf, -np.inf], np.nan)
    eligible = eligible.dropna(subset=["annual_inc", "fico_range_low", "loan_amnt", "target"])
    eligible = eligible[
        (eligible["annual_inc"] > 0)
        & (eligible["loan_amnt"] > 0)
        & (eligible["fico_range_low"] >= 300)
    ]
    if eligible["target"].nunique() < 2 or len(eligible) < 500:
        raise ValueError("There are not enough resolved historical loans to produce an assessment.")

    if len(eligible) > MAX_TRAINING_ROWS:
        eligible = eligible.sample(n=MAX_TRAINING_ROWS, random_state=42)

    x = eligible[FEATURES].astype("float64")
    y = eligible["target"].astype("int8")
    estimator = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=500, solver="lbfgs", C=0.8)),
        ]
    )
    estimator.fit(x, y)
    return estimator


def risk_band(probability: float) -> str:
    value = float(np.clip(probability, 0.0, 1.0))
    if value < 0.05:
        return "Lower"
    if value < 0.10:
        return "Moderate"
    if value < 0.20:
        return "Elevated"
    if value < 0.35:
        return "High"
    return "Very high"


def peer_benchmark(
    portfolio: pd.DataFrame,
    annual_income: float,
    dti: float,
    fico: float,
    loan_amount: float,
) -> dict[str, Any]:
    """Return historical observed outcomes for a broad comparable-borrower cohort."""
    resolved = portfolio.loc[portfolio["target"].notna()]
    if resolved.empty:
        return {"count": 0, "default_rate": None}

    base = (
        (resolved["fico_range_low"].between(max(300, fico - 30), min(850, fico + 30)))
        & (resolved["dti"].between(max(0, dti - 8), dti + 8))
        & (resolved["annual_inc"].between(max(1_000, annual_income * 0.5), annual_income * 1.5))
        & (resolved["loan_amnt"].between(max(500, loan_amount * 0.5), loan_amount * 1.5))
    )
    peers = resolved.loc[base, "target"]
    if len(peers) < 50:
        broader = (
            (resolved["fico_range_low"].between(max(300, fico - 45), min(850, fico + 45)))
            & (resolved["dti"].between(max(0, dti - 12), dti + 12))
            & (resolved["loan_amnt"].between(max(500, loan_amount * 0.35), loan_amount * 1.75))
        )
        peers = resolved.loc[broader, "target"]

    return {
        "count": int(len(peers)),
        "default_rate": float(peers.mean()) if len(peers) else None,
    }


def format_period_range(portfolio: pd.DataFrame) -> str:
    years = portfolio["issue_year"].dropna()
    if years.empty:
        return "Period unavailable"
    return f"{int(years.min())}–{int(years.max())}"
