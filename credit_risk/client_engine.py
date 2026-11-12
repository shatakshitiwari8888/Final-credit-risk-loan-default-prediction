"""Client-facing LendingClub portfolio data and borrower assessment utilities.

This module intentionally exposes business outputs only. Model fitting and
preprocessing remain behind the dashboard interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import os
import time

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
MAX_TRAINING_ROWS = 50_000
LARGE_ARCHIVE_BYTES = 100 * 1024 * 1024
SAMPLE_FRACTION = 0.04
READ_CHUNK_ROWS = 40_000
logger = logging.getLogger(__name__)


def log_memory(stage: str) -> None:
    """Report process RSS and high-water mark to Streamlit deployment logs."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            values = dict(line.split(":", 1) for line in fh if ":" in line)
        logger.warning("portfolio_memory stage=%s pid=%s rss=%s peak=%s",
                       stage, os.getpid(), values.get("VmRSS", "").strip(),
                       values.get("VmHWM", "").strip())
    except (OSError, ValueError):
        logger.info("portfolio_memory stage=%s (RSS unavailable)", stage)



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

    started = time.monotonic()
    log_memory("load_start")
    use_sample = path.stat().st_size >= LARGE_ARCHIVE_BYTES
    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    available = [column for column in DISPLAY_FIELDS if column in header]
    required = {"loan_status", "loan_amnt", "annual_inc", "dti", "fico_range_low"}
    missing = sorted(required - set(available))
    if missing:
        raise ValueError(
            "The uploaded file is missing required portfolio fields: " + ", ".join(missing)
        )

    chunks: list[pd.DataFrame] = []
    rows_scanned = 0
    for index, chunk in enumerate(pd.read_csv(
        path,
        compression="gzip",
        usecols=available,
        chunksize=READ_CHUNK_ROWS,
        low_memory=False,
        dtype_backend="pyarrow",
    )):
        rows_scanned += len(chunk)
        if use_sample:
            # Retain a deterministic 4% sample rather than all 2.26M records.
            # Per-chunk sampling bounds resident data during CSV parsing.
            chunk = chunk.sample(n=max(1, round(len(chunk) * SAMPLE_FRACTION)),
                                 random_state=42 + index)
        chunks.append(chunk)
        if index % 10 == 0:
            log_memory(f"read_chunk_{index}")
    frame = pd.concat(chunks, ignore_index=True, copy=False) if chunks else pd.DataFrame(columns=available)
    del chunks
    logger.warning("portfolio_load source_rows=%s retained_rows=%s sampled=%s elapsed_s=%.1f",
                   rows_scanned, len(frame), use_sample, time.monotonic() - started)
    log_memory("after_concat")

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
    frame.attrs["sampled"] = use_sample
    frame.attrs["source_rows"] = rows_scanned
    log_memory("load_complete")
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

    log_memory("training_start")
    eligible = portfolio.loc[
        portfolio["target"].notna(),
        ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti", "target"],
    ]
    # Sample before the copies required by sklearn conversion and imputation.
    if len(eligible) > MAX_TRAINING_ROWS:
        eligible = eligible.sample(n=MAX_TRAINING_ROWS, random_state=42)
    eligible = eligible.copy()
    eligible = eligible.replace([np.inf, -np.inf], np.nan)
    eligible = eligible.dropna(subset=["annual_inc", "fico_range_low", "loan_amnt", "target"])
    eligible = eligible[
        (eligible["annual_inc"] > 0)
        & (eligible["loan_amnt"] > 0)
        & (eligible["fico_range_low"] >= 300)
    ]
    if eligible["target"].nunique() < 2 or len(eligible) < 500:
        raise ValueError("There are not enough resolved historical loans to produce an assessment.")

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
    log_memory("training_complete")
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


def historical_grade_assessment(portfolio: pd.DataFrame, probability: float) -> dict[str, Any]:
    """Align modeled default risk with observed LendingClub origination grades.

    This is a descriptive, indicative grade, NOT a lender-issued credit grade
    or a lending/approval decision. Only sufficiently represented resolved
    historical loans are considered.
    """
    grades = list("ABCDEFG")
    valid = portfolio.loc[
        portfolio["target"].notna() & portfolio["grade"].isin(grades), ["grade", "target"]
    ]
    if valid.empty:
        return {"grade": "Unrated", "reference": pd.DataFrame()}
    reference = (
        valid.groupby("grade", observed=True)["target"]
        .agg(loans="size", observed_rate="mean")
        .reindex(grades)
        .dropna()
    )
    reference = reference.loc[reference["loans"] >= 30].copy()
    if len(reference) < 2:
        return {"grade": "Unrated", "reference": reference}
    # Maintain a non-decreasing grade-risk ordering despite sample noise.
    ordered_rates = np.maximum.accumulate(reference["observed_rate"].to_numpy(dtype=float))
    cutoffs = (ordered_rates[:-1] + ordered_rates[1:]) / 2.0
    grade = str(reference.index[int(np.searchsorted(cutoffs, probability, side="right"))])
    return {"grade": grade, "reference": reference}


def logistic_contributions(estimator: Any, features: pd.DataFrame) -> dict[str, Any]:
    """Exact per-feature standardized log-odds decomposition of fitted model."""
    imputed = estimator.named_steps["impute"].transform(features[FEATURES])
    standardized = estimator.named_steps["scale"].transform(imputed)[0]
    model = estimator.named_steps["logistic"]
    coefficients = model.coef_[0]
    contributions = standardized * coefficients
    intercept = float(model.intercept_[0])
    log_odds = intercept + float(contributions.sum())
    rows = pd.DataFrame({
        "Feature": ["Annual income", "Debt-to-income", "FICO score",
                    "Loan amount", "Loan-to-income"],
        "Input": imputed[0],
        "Z-score": standardized,
        "Coefficient": coefficients,
        "Log-odds contribution": contributions,
    })
    return {"rows": rows, "intercept": intercept, "log_odds": log_odds}


def borrower_sensitivity(estimator: Any, income: float, dti: float,
                         fico: float, loan_amount: float,
                         income_gain: float = 0.0, dti_reduction: float = 0.0,
                         fico_gain: float = 0.0) -> dict[str, float]:
    """Model-driven borrower what-if; does not promise a credit decision."""
    baseline = prepare_borrower_features(income, dti, fico, loan_amount)
    changed = prepare_borrower_features(
        income + max(0.0, income_gain),
        max(0.0, dti - max(0.0, dti_reduction)),
        min(850.0, fico + max(0.0, fico_gain)),
        loan_amount,
    )
    return {
        "baseline_pd": float(estimator.predict_proba(baseline)[0, 1]),
        "simulated_pd": float(estimator.predict_proba(changed)[0, 1]),
        "simulated_income": float(changed.iloc[0]["annual_inc"]),
        "simulated_dti": float(changed.iloc[0]["dti"]),
        "simulated_fico": float(changed.iloc[0]["fico_range_low"]),
    }


def stress_test_sample(portfolio: pd.DataFrame, estimator: Any,
                       income_decline_pct: float = 0.0,
                       dti_increase_points: float = 0.0,
                       lgd: float = 0.45,
                       max_rows: int = 5000) -> dict[str, float]:
    """Illustrative EAD×PD×LGD shock on a bounded sample of historical loans.

    The same rows under baseline and stress are scored by the fitted model.
    This is NOT a forecast, a regulatory stress test, or full-book loss.
    """
    if not (0.0 <= lgd <= 1.0):
        raise ValueError("LGD must be between 0 and 1")
    eligible = portfolio.loc[
        portfolio["annual_inc"].gt(0) & portfolio["loan_amnt"].gt(0)
        & portfolio["fico_range_low"].notna(),
        ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti"],
    ]
    if eligible.empty:
        raise ValueError("No valid borrower profiles for scenario evaluation")
    if len(eligible) > max_rows:
        eligible = eligible.sample(n=max_rows, random_state=52)
    x = eligible.astype("float64").copy()
    base_pd = estimator.predict_proba(x)[:, 1]
    scenario = x.copy()
    scenario["annual_inc"] = (scenario["annual_inc"]
                              * (1.0 - max(0.0, min(95.0, income_decline_pct)) / 100.0))
    scenario["dti"] = (scenario["dti"] + max(0.0, dti_increase_points)).clip(upper=100)
    scenario["lti"] = (scenario["loan_amnt"]
                        / scenario["annual_inc"].clip(lower=1000)).clip(upper=5)
    stressed_pd = estimator.predict_proba(scenario)[:, 1]
    ead = x["loan_amnt"].to_numpy()
    base_el = float(np.dot(ead, base_pd) * lgd)
    stressed_el = float(np.dot(ead, stressed_pd) * lgd)
    return {
        "loans": int(len(x)),
        "exposure": float(ead.sum()),
        "baseline_pd": float(base_pd.mean()),
        "stressed_pd": float(stressed_pd.mean()),
        "baseline_el": base_el,
        "stressed_el": stressed_el,
        "incremental_el": stressed_el - base_el,
    }


def evaluate_borrower_model(portfolio: pd.DataFrame,
                            max_rows: int = 15000) -> dict[str, Any]:
    """Train/holdout metrics from a reproducible, stratified 80/20 partition.

    Separate from the production estimator: all displayed accuracy statistics
    are evaluated on loans that were not used for training this diagnostic model.
    """
    from sklearn.base import clone
    from sklearn.metrics import (accuracy_score, brier_score_loss,
        confusion_matrix, f1_score, precision_score, recall_score,
        roc_auc_score, roc_curve)
    from sklearn.model_selection import train_test_split

    cols = FEATURES + ["target"]
    eligible = portfolio.loc[portfolio["target"].notna(), cols]
    eligible = eligible.loc[
        eligible["annual_inc"].gt(0) & eligible["loan_amnt"].gt(0)
        & eligible["fico_range_low"].ge(300)
    ]
    if len(eligible) > max_rows:
        eligible = eligible.sample(n=max_rows, random_state=71)
    if len(eligible) < 500 or eligible["target"].nunique() < 2:
        raise ValueError("Not enough labeled resolved loans for holdout validation")
    x = eligible[FEATURES].astype("float64").replace([np.inf, -np.inf], np.nan)
    y = eligible["target"].astype("int8")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y,
    )
    # Use the exact same model family and preprocessing as the live app.
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=500, solver="lbfgs", C=0.8)),
    ])
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = 0.30
    predicted = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    calibration = pd.DataFrame({"pred": probabilities, "actual": y_test.to_numpy()})
    calibration["bin"] = pd.qcut(calibration["pred"], q=8, duplicates="drop")
    calibration = calibration.groupby("bin", observed=True).agg(
        predicted_pd=("pred", "mean"),
        observed_default=("actual", "mean"),
        loans=("actual", "size"),
    ).reset_index(drop=True)
    return {
        "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
        "auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": fpr, "tpr": tpr, "calibration": calibration,
        "threshold": threshold,
    }
