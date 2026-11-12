import numpy as np
import pandas as pd

from credit_risk.client_engine import (
    fit_borrower_model,
    format_period_range,
    peer_benchmark,
    prepare_borrower_features,
    risk_band,
)


def sample_portfolio(rows: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    annual_inc = rng.normal(85000, 25000, rows).clip(18000, 250000)
    dti = rng.normal(18, 8, rows).clip(0, 55)
    fico = rng.normal(700, 45, rows).clip(560, 850)
    loan_amnt = rng.normal(16000, 7000, rows).clip(1000, 45000)
    lti = loan_amnt / annual_inc
    linear = -2.4 - 0.012 * (fico - 700) + 0.04 * (dti - 18) + 1.2 * (lti - 0.2)
    probability = 1 / (1 + np.exp(-linear))
    target = (rng.random(rows) < probability).astype(float)
    return pd.DataFrame(
        {
            "annual_inc": annual_inc,
            "dti": dti,
            "fico_range_low": fico,
            "loan_amnt": loan_amnt,
            "lti": lti,
            "target": target,
            "issue_year": rng.integers(2010, 2019, rows),
        }
    )


def test_borrower_feature_preparation():
    row = prepare_borrower_features(80000, 22, 710, 18000)
    assert list(row.columns) == ["annual_inc", "dti", "fico_range_low", "loan_amnt", "lti"]
    assert row.loc[0, "lti"] > 0


def test_risk_bands_are_ordered_and_bounded():
    assert risk_band(0.02) == "Lower"
    assert risk_band(0.08) == "Moderate"
    assert risk_band(0.15) == "Elevated"
    assert risk_band(0.27) == "High"
    assert risk_band(0.50) == "Very high"


def test_borrower_model_and_peer_benchmark():
    data = sample_portfolio()
    model = fit_borrower_model(data)
    applicant = prepare_borrower_features(85000, 18, 700, 16000)
    probability = float(model.predict_proba(applicant)[0, 1])
    assert 0 <= probability <= 1

    peers = peer_benchmark(data, 85000, 18, 700, 16000)
    assert peers["count"] > 0
    assert peers["default_rate"] is not None
    assert 0 <= peers["default_rate"] <= 1


def test_period_range():
    assert format_period_range(sample_portfolio()) == "2010–2018"


def test_large_archive_sampling_is_bounded(tmp_path, monkeypatch):
    from credit_risk import client_engine
    frame = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off"] * 2500,
        "loan_amnt": [15000] * 5000,
        "annual_inc": [75000] * 5000,
        "dti": [17] * 5000,
        "fico_range_low": [700] * 5000,
    })
    path = tmp_path / "loans.csv.gz"
    frame.to_csv(path, compression="gzip", index=False)
    monkeypatch.setattr(client_engine, "LARGE_ARCHIVE_BYTES", 1)
    result = client_engine.load_portfolio(path)
    assert len(result) == 200
    assert result.attrs["source_rows"] == 5000
    assert result.attrs["sampled"] is True
    assert set(result["outcome"].astype(str)) == {"Fully paid", "Charged off"}


def test_small_archive_keeps_all_rows(tmp_path):
    from credit_risk.client_engine import load_portfolio
    frame = pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off"],
        "loan_amnt": [10000, 20000],
        "annual_inc": [50000, 60000],
        "dti": [15, 20],
        "fico_range_low": [710, 680],
    })
    path = tmp_path / "small.csv.gz"
    frame.to_csv(path, compression="gzip", index=False)
    result = load_portfolio(path)
    assert len(result) == 2
    assert result.attrs["sampled"] is False


def test_indicative_grade_uses_historical_origination_grade_rates():
    from credit_risk.client_engine import historical_grade_assessment
    rows = []
    for grade, defaults in [("A", 2), ("B", 10), ("C", 20), ("D", 32),
                            ("E", 48), ("F", 62), ("G", 78)]:
        for i in range(100):
            rows.append({"grade": grade, "target": float(i < defaults)})
    portfolio = pd.DataFrame(rows)
    grades = [historical_grade_assessment(portfolio, p)["grade"]
              for p in (0.02, 0.10, 0.20, 0.32, 0.48, 0.62, 0.78)]
    assert grades == list("ABCDEFG")
    assert historical_grade_assessment(portfolio, 0.98)["grade"] == "G"


def test_logistic_decomposition_reconstructs_probability():
    from credit_risk.client_engine import logistic_contributions
    dataset = sample_portfolio()
    model = fit_borrower_model(dataset)
    sample = prepare_borrower_features(75000, 18, 700, 15000)
    info = logistic_contributions(model, sample)
    log_odds = info["intercept"] + float(info["rows"]["Log-odds contribution"].sum())
    assert abs(log_odds - info["log_odds"]) < 1e-9
    reconstructed = 1 / (1 + np.exp(-log_odds))
    predicted = model.predict_proba(sample)[0, 1]
    assert abs(reconstructed - predicted) < 1e-9


def test_borrower_what_if_and_stress_scenarios_are_model_based():
    from credit_risk.client_engine import borrower_sensitivity, stress_test_sample
    portfolio = sample_portfolio(1500)
    model = fit_borrower_model(portfolio)
    scenario = borrower_sensitivity(model, 75000, 18, 700, 15000,
                                    income_gain=10000, dti_reduction=5, fico_gain=20)
    assert 0 <= scenario["baseline_pd"] <= 1
    assert 0 <= scenario["simulated_pd"] <= 1
    assert scenario["simulated_fico"] == 720
    base = stress_test_sample(portfolio, model, 0, 0, lgd=.45, max_rows=500)
    stressed = stress_test_sample(portfolio, model, 20, 10, lgd=.45, max_rows=500)
    assert base["loans"] == 500
    assert abs(base["baseline_el"] - base["stressed_el"]) < 1e-6
    assert stressed["stressed_el"] >= 0
    assert stressed["loans"] == 500


def test_holdout_diagnostics_are_bounded_and_reproducible():
    from credit_risk.client_engine import evaluate_borrower_model
    portfolio = sample_portfolio(1800)
    results = evaluate_borrower_model(portfolio, max_rows=1000)
    assert results["train_rows"] + results["test_rows"] == 1000
    assert 0 <= results["auc"] <= 1
    assert 0 <= results["brier"] <= 1
    assert results["tn"] + results["fp"] + results["fn"] + results["tp"] == results["test_rows"]
    assert not results["calibration"].empty
