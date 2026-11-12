# Loan Portfolio Intelligence

A Python + Streamlit dashboard for client-facing credit portfolio analysis. The application presents portfolio exposure, historical repayment outcomes, risk segmentation, and an indicative borrower assessment.

## Client dashboard

The dashboard is intentionally limited to three workspaces:

1. **Portfolio Summary** — loan volume, funded exposure, observed charge-off rate, portfolio profile and origination trends.
2. **Risk & Segments** — outcomes across origination grade, debt-to-income, FICO, loan term, purpose, state and origination year.
3. **Borrower Assessment** — an indicative probability of default and a historical comparable-borrower benchmark.

The dashboard does not expose source code, model coefficients, training procedures, developer notes or academic defence material.

## Data source

The dashboard is designed for LendingClub accepted-loan records from 2007 through 2018 Q4. The supplied archive is approximately 392.6 MB compressed. The application reads only the loan-level fields required for portfolio analysis and borrower assessment; it does not display borrower names, account identifiers, street addresses or other direct identifiers.

The charge-off rate is calculated only for loans with terminal outcomes:
- `Fully Paid` → repaid
- `Charged Off` / `Default` → charged off
- Current, late, grace-period and other unresolved loans are excluded from the resolved-outcome denominator.

The borrower assessment is an indicative historical-data-based estimate. It is not an automated approval, decline or pricing decision.

## Add the raw dataset

GitHub blocks regular Git files above 100 MiB. This dataset must be stored using Git LFS. Streamlit Community Cloud supports Git LFS-backed files.

From a local clone of this repository:

```bash
git lfs install
git lfs track "data/accepted_2007_to_2018Q4.csv.gz"
```

Copy the supplied `accepted_2007_to_2018Q4.csv.gz` file into the repository's `data/` directory, then run:

```bash
git add .gitattributes .gitignore data/accepted_2007_to_2018Q4.csv.gz
git commit -m "Add LendingClub accepted-loan portfolio data"
git push origin client-presentation
```

Once the large file has been pushed through Git LFS, merge `client-presentation` into `main` (or deploy the `client-presentation` branch). Streamlit Community Cloud will then retrieve the LFS object when it deploys the app.

## Run locally

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Runtime

- Python
- Streamlit
- Pandas / NumPy
- scikit-learn
- Plotly

All application and analytics code is Python. No React, TypeScript, JavaScript or Node.js frontend is used.

## Responsible interpretation

Historical charge-off rates describe observed outcomes in the supplied dataset. They are not forecasts. Borrower estimates are indicative decision-support information and should be reviewed alongside verified application documents, bureau data, affordability checks, lender policy and applicable compliance requirements.
