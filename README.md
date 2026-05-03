# Unobserved Performance of Hedge Funds

**Replication of Weigert, Wegener & Klesczewski (*Journal of Finance*, 2024)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository provides a complete Python replication of the empirical methodology from:

> Weigert, F., Wegener, C., & Klesczewski, E. (2024). **Unobserved Performance of Hedge Funds.** *Journal of Finance*, 79(4), 2399–2452.

The paper introduces **Unobserved Performance (UP)**, a measure defined as the difference between a hedge fund's actual reported return and the hypothetical buy-and-hold return implied by its most recently disclosed equity holdings (13F filings). UP captures value generated through channels invisible to standard holdings-based analysis:

- **Non-equity positions** — fixed income, currencies, commodities, derivatives
- **Intra-quarter trading** — active management between quarterly disclosure dates
- **Short positions** — not captured in mandatory 13F long-equity filings

The central empirical finding is that UP positively and significantly predicts future fund returns: a long-short portfolio that buys funds in the highest UP quintile and sells funds in the lowest UP quintile generates economically and statistically significant risk-adjusted alpha.

---
```bash
## Repository Structure

hedge-fund-up/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── replicate_up.py # Main replication script
├── data/
│ ├── UP_5-1.csv # Published portfolio sort results (validation)
│ └── results/ # Output directory (auto-created)
│ ├── up_panel.csv
│ ├── long_short_returns.csv
│ └── quintile_returns.csv
└── notebooks/
└── analysis.ipynb # Optional exploratory notebook
```
## Methodology

The replication pipeline proceeds in five stages:

### Stage 1 — Data Preparation

Monthly fund returns, quarterly 13F equity holdings, stock returns, and Fama-French + Momentum factors are loaded from CSV files (or generated synthetically in demo mode). Monthly returns are geometrically compounded to quarterly frequency to match the holdings disclosure calendar.

### Stage 2 — UP Construction

For each fund-quarter observation:

$$
\text{UP}_{i,t} = R^{\text{reported}}_{i,t} - R^{\text{buy-and-hold}}_{i,t}
$$

where the buy-and-hold return applies the most recently disclosed portfolio weights to realized stock returns during quarter *t*. The computation uses a pre-indexed holdings lookup for performance (O(1) per fund rather than O(N) filtering).

### Stage 3 — Predictive Portfolio Sorts

Each quarter *t*, funds are sorted cross-sectionally into quintile portfolios by UP(*t*). Equal-weighted portfolio returns are then measured at horizon *t+k* (default *k* = 1 quarter; the paper's main specification uses monthly data with *t+3* months).

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

Standard errors are estimated using the Newey-West (1987) HAC covariance matrix with 4 lags to account for serial correlation and heteroskedasticity.

---

## Installation

### Requirements

- Python ≥ 3.9
- pandas ≥ 1.5
- numpy ≥ 1.22
- statsmodels ≥ 0.13

### Setup

```bash
git clone https://github.com/your-username/hedge-fund-up.git
cd hedge-fund-up
pip install -r requirements.txt
```
