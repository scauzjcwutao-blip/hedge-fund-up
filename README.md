# Unobserved Performance of Hedge Funds

**Replication of AGARWAL, V., RUENZI, S. and WEIGERT, F.  (*Journal of Finance*, 2024)**

[![Python 3.9+](https://pfst.cf2.poecdn.net/base/image/c3246e60e0df128e5a23387e31f19ce1a47d45a94136c1fb2e6931cdfd87c9db?pmaid=616215989)](https://www.python.org/downloads/)
[![License: MIT](https://pfst.cf2.poecdn.net/base/image/4ba58a42a1fd9162b3759f934f77fea471b851e359255d522e10f301557e22c3?pmaid=616215988)](LICENSE)

---

## Overview

This repository provides a complete Python replication of the empirical methodology from:

> AGARWAL, V., RUENZI, S. and WEIGERT, F. (2024), **Unobserved Performance of Hedge Funds**. J Finance, 79: 3203-3259. https://doi.org/10.1111/jofi.13368

The paper introduces **Unobserved Performance (UP)**, a measure defined as the difference between a hedge fund's actual reported return and the hypothetical buy-and-hold return implied by its most recently disclosed equity holdings (13F filings). UP captures value generated through channels invisible to standard holdings-based analysis:

- **Non-equity positions** — fixed income, currencies, commodities, derivatives
- **Intra-quarter trading** — active management between quarterly disclosure dates
- **Short positions** — not captured in mandatory 13F long-equity filings

The central empirical finding is that UP positively and significantly predicts future fund returns: a long-short portfolio that buys funds in the highest UP quintile and sells funds in the lowest UP quintile generates economically and statistically significant risk-adjusted alpha.

## Key Features

- **Dual frequency support** — run the analysis at monthly or quarterly granularity via `--freq`
- Full multi-fund panel processing with correct cross-sectional quintile sorting
- Monthly UP computation with quarterly holdings interpolated to each month (matching the paper)
- Quarterly UP computation with compounded returns (computationally efficient approximation)
- Predictive long-short portfolios with configurable horizon (`--horizon`)
- Carhart 4-factor alphas with Newey-West HAC standard errors (lags auto-adjusted by frequency)
- Fama-MacBeth persistence tests
- Built-in validation against the paper's official UP 5-1 benchmark
- Clean command-line interface (demo + real WRDS data)

---

## Frequency Modes

The script supports two analysis frequencies selected via the `--freq` flag:

### Monthly Mode (`--freq monthly`)

This mode closely replicates the paper's primary specification. UP is computed at the monthly level: each month, the most recent quarterly 13F disclosure is used to construct a hypothetical buy-and-hold return, which is subtracted from the fund's actual monthly return. Cross-sectional quintile sorts are performed monthly, and evaluation returns are measured at t+3 months (default) to match the paper's main results.

| Parameter | Default |
|-----------|---------|
| Horizon | t + 3 months |
| Newey-West lags | 6 |
| Annualization | × 12 |

### Quarterly Mode (`--freq quarterly`, default)

This mode compounds monthly returns to quarterly frequency before computing UP, sorting, and forming portfolios. Because 13F holdings are disclosed quarterly, this approach naturally aligns holdings with evaluation periods. It is computationally faster and preserves the economic intuition of the original study, though it operates at lower time-series granularity.

| Parameter | Default |
|-----------|---------|
| Horizon | t + 1 quarter |
| Newey-West lags | 4 |
| Annualization | × 4 |

The built-in `validate_against_paper()` function automatically handles the comparison: in monthly mode it compares directly against the paper's monthly benchmark; in quarterly mode it uses a ×3 linear approximation for illustration.

---
```bash
## Repository Structure
hedge-fund-up/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── replicate_up.py # Main replication script (v3.0)
├── data/
│ ├─── UP 5-1 (1).xls  # Published portfolio sort results (validation)
│ └── results/ # Output directory (auto-created)
│ ├── up_panel_monthly.csv
│ ├── up_panel_quarterly.csv
│ ├── long_short_returns_monthly.csv
│ ├── long_short_returns_quarterly.csv
│ ├── quintile_returns_monthly.csv
│ └── quintile_returns_quarterly.csv
└── notebooks/
└── analysis.ipynb # Optional exploratory notebook
```

## Methodology

The replication pipeline proceeds in seven stages:

### Stage 1 — Data Preparation

Monthly fund returns, quarterly 13F equity holdings, stock returns, and Fama-French + Momentum factors are loaded from CSV files (or generated synthetically in demo mode).

In **quarterly mode**, monthly returns and factors are geometrically compounded to quarterly frequency to match the holdings disclosure calendar. In **monthly mode**, stock returns are pivoted into a monthly wide-format matrix and fund returns are used directly at monthly granularity.

### Stage 2 — UP Construction

For each fund-period observation:

$$
\text{UP}_{i,t} = R^{\text{reported}}_{i,t} - R^{\text{buy-and-hold}}_{i,t}
$$

where the buy-and-hold return applies the most recently disclosed portfolio weights (from the latest 13F filing on or before period *t*) to realized stock returns during period *t*.

In **monthly mode**, the same quarterly holdings are applied to each month within (and beyond) that quarter until the next disclosure arrives, capturing the "staleness" of publicly observable information — exactly as in the paper's construction.

### Stage 3 — Predictive Portfolio Sorts

Each period *t*, funds are sorted cross-sectionally into quintile portfolios by UP(*t*). Equal-weighted portfolio returns are then measured at horizon *t + k*:

- Monthly mode default: *k* = 3 months (paper's main specification)
- Quarterly mode default: *k* = 1 quarter

### Stage 4 — Long-Short Strategy

The long-short spread is defined as:

$$
R^{LS}_t = R^{Q5}_t - R^{Q1}_t
$$

where Q5 contains funds with the highest UP (strongest unobserved skill) and Q1 contains funds with the lowest UP.

### Stage 5 — Risk-Adjusted Alpha

The long-short return series is regressed on the Carhart (1997) four-factor model:

$$
R^{LS}_t = \alpha + \beta_1 \text{MKT}_t + \beta_2 \text{SMB}_t + \beta_3 \text{HML}_t + \beta_4 \text{MOM}_t + \varepsilon_t
$$

Standard errors are estimated using the Newey-West (1987) HAC covariance matrix (6 lags for monthly, 4 lags for quarterly) to account for serial correlation and heteroskedasticity.

### Stage 6 — Quintile Monotonicity Analysis

Individual Carhart 4-factor alphas are estimated for each quintile portfolio to verify the monotonic increase from Q1 to Q5, as documented in the paper.

### Stage 7 — UP Persistence (Fama-MacBeth)

Fama-MacBeth cross-sectional regressions test whether UP(*t−1*) predicts UP(*t*), providing evidence on whether unobserved performance reflects persistent managerial skill.
```bash
# DEMO——Quarterly mode (default) — fast, good for testing
python replicate_up.py --demo --freq quarterly

# Monthly mode — matches the paper's specification
python replicate_up.py --demo --freq monthly --horizon 3

# Monthly with validation against published results
python replicate_up.py --demo --freq monthly --horizon 3 --validate data/UP_5-1.xls
---
# DEMO——Monthly frequency replicating the paper's t+3 specification with Real WRDS Data
python replicate_up.py --freq monthly --horizon 3 \
    --fund-returns data/fund_returns.csv \
    --holdings data/holdings.csv \
    --stock-returns data/stock_returns.csv \
    --factors data/factors.csv \
    --validate data/UP_5-1.xls

# Quarterly approximation
python replicate_up.py --freq quarterly --horizon 1 \
    --fund-returns data/fund_returns.csv \
    --holdings data/holdings.csv \
    --stock-returns data/stock_returns.csv \
    --factors data/factors.csv \
    --validate data/UP_5-1.xls
## Installation

### Requirements

- Python ≥ 3.9 (tested up to 3.14)
- pandas ≥ 1.5
- numpy ≥ 1.22
- statsmodels ≥ 0.13

### Setup

```bash
git clone https://github.com/scauzjcwutao-blip/hedge-fund-up.git
cd hedge-fund-up
pip install -r requirements.txt
# Quarterly mode (default) — fast, good for testing
python replicate_up.py --demo --freq quarterly

# Monthly mode — matches the paper's specification
python replicate_up.py --demo --freq monthly --horizon 3

# Monthly with validation against published results
python replicate_up.py --demo --freq monthly --horizon 3 --validate data/UP_5-1.csv
@article{agarwal2024unobserved,
  title={Unobserved Performance of Hedge Funds},
  author={Agarwal, Vikas and Ruenzi, Stefan and Weigert, Florian},
  journal={The Journal of Finance},
  volume={79},
  number={5},
  pages={3203--3259},
  year={2024},
  publisher={Wiley},
  doi={10.1111/jofi.13368}
}
