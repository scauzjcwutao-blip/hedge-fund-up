"""
================================================================================
Unobserved Performance (UP) of Hedge Funds
Replication of Weigert, Wegener, and Klesczewski (Journal of Finance, 2024)
================================================================================

Author:  Tao Wu
Date:    May 2026
License: MIT

Description:
    This script replicates the core empirical methodology of "Unobserved
    Performance of Hedge Funds" (Journal of Finance, 2024). The paper proposes
    a measure called Unobserved Performance (UP), defined as the difference
    between a hedge fund's actual reported return and the hypothetical
    buy-and-hold return implied by its most recently disclosed equity holdings.

    UP captures value generated through:
      - Non-equity positions (fixed income, derivatives, etc.)
      - Intra-quarter trading (active management between disclosure dates)
      - Short positions (not captured in 13F filings)

    The key finding is that UP positively predicts future fund returns:
    a long-short portfolio buying high-UP funds and selling low-UP funds
    generates significant risk-adjusted alpha.

Methodology:
    1. Construct UP(t) = R_fund(t) - R_buyandhold(t) for each fund-period.
    2. Each period t, sort funds into quintile portfolios by UP(t).
    3. Track equal-weighted portfolio returns in period t+k (k=1 or k=3).
    4. The long-short spread (Q5 - Q1) is regressed on Carhart four factors
       using Newey-West (HAC) standard errors to obtain risk-adjusted alpha.

Data Requirements (for full replication with WRDS data):
    - fund_returns.csv:  columns ['date', 'fund_id', 'fund_ret']
    - holdings.csv:      columns ['report_date', 'fund_id', 'stock', 'weight']
    - stock_returns.csv: columns ['date', 'stock', 'ret']
    - factors.csv:       columns ['date', 'MKT', 'SMB', 'HML', 'MOM']

    All returns should be in decimal form (e.g., 0.05 for 5%).
    Dates should be parseable by pandas (e.g., YYYY-MM-DD).

Usage:
    # Run with demo data (no external files needed):
    python replicate_up.py --demo

    # Run with real WRDS data:
    python replicate_up.py \\
        --fund-returns data/fund_returns.csv \\
        --holdings data/holdings.csv \\
        --stock-returns data/stock_returns.csv \\
        --factors data/factors.csv

    # Validate against published results:
    python replicate_up.py --demo --validate data/UP_5-1.csv

    # Customize prediction horizon (default=3 months as in paper):
    python replicate_up.py --demo --horizon 3

References:
    Weigert, F., Wegener, C., & Klesczewski, E. (2024).
    Unobserved Performance of Hedge Funds.
    Journal of Finance, 79(4), 2399-2452.
================================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__version__ = "1.0.0"


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_SEED = 42
MIN_MONTHS_PER_QUARTER = 2
MIN_FUNDS_PER_PERIOD = 20
N_QUINTILES = 5
NEWEY_WEST_LAGS = 4
FACTOR_COLS = ["MKT", "SMB", "HML", "MOM"]


# =============================================================================
# SECTION 1: DATA GENERATION (DEMO MODE)
# =============================================================================

def generate_demo_data(
    n_funds: int = 100,
    n_stocks: int = 50,
    start: str = "1994-01-31",
    end: str = "2019-12-31",
    seed: int = DEFAULT_SEED,
) -> tuple:
    """
    Generate synthetic panel data for testing the replication pipeline.

    The demo data mimics the structure of real hedge fund data with:
    - Heterogeneous fund alphas and volatilities
    - Quarterly 13F-style holdings reports
    - Correlated stock returns
    - Standard Fama-French + Momentum factors

    Parameters
    ----------
    n_funds : int
        Number of simulated hedge funds.
    n_stocks : int
        Number of simulated stocks in the universe.
    start, end : str
        Date range for the simulation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of DataFrames
        (fund_returns_df, holdings_df, stock_returns_df, factors_df)
    """
    print(f"\n{'─'*60}")
    print(f"  Generating Demo Data: {n_funds} funds, {n_stocks} stocks")
    print(f"  Period: {start} to {end}")
    print(f"{'─'*60}")

    rng = np.random.default_rng(seed)

    # Use legacy freq strings for broad pandas compatibility
    try:
        months = pd.date_range(start=start, end=end, freq="ME")
    except ValueError:
        months = pd.date_range(start=start, end=end, freq="M")

    try:
        quarters = pd.date_range(start="1994-03-31", end="2019-12-31", freq="QE")
    except ValueError:
        quarters = pd.date_range(start="1994-03-31", end="2019-12-31", freq="Q")

    fund_ids = [f"FUND{i:04d}" for i in range(1, n_funds + 1)]
    stock_ids = [f"STK{i:03d}" for i in range(1, n_stocks + 1)]

    # --- Fund Monthly Returns ---
    fund_records = []
    for fid in fund_ids:
        alpha = rng.normal(0.005, 0.003)
        vol = rng.uniform(0.02, 0.06)
        rets = rng.normal(alpha, vol, len(months))
        for date, ret in zip(months, rets):
            fund_records.append({"date": date, "fund_id": fid, "fund_ret": ret})
    fund_returns_df = pd.DataFrame(fund_records)

    # --- Stock Monthly Returns ---
    stock_records = []
    for sid in stock_ids:
        mu = rng.normal(0.008, 0.005)
        vol = rng.uniform(0.04, 0.12)
        rets = rng.normal(mu, vol, len(months))
        for date, ret in zip(months, rets):
            stock_records.append({"date": date, "stock": sid, "ret": ret})
    stock_returns_df = pd.DataFrame(stock_records)

    # --- Quarterly Holdings ---
    holdings_records = []
    for fid in fund_ids:
        n_hold = rng.integers(5, 20)
        current_stocks = rng.choice(stock_ids, n_hold, replace=False)
        for q in quarters:
            if rng.random() < 0.3:
                n_hold = rng.integers(5, 20)
                current_stocks = rng.choice(stock_ids, n_hold, replace=False)
            weights = rng.dirichlet(np.ones(len(current_stocks)))
            for stock, weight in zip(current_stocks, weights):
                holdings_records.append(
                    {"report_date": q, "fund_id": fid, "stock": stock, "weight": weight}
                )
    holdings_df = pd.DataFrame(holdings_records)

    # --- Monthly Factors ---
    factors_df = pd.DataFrame(
        {
            "date": months,
            "MKT": rng.normal(0.006, 0.04, len(months)),
            "SMB": rng.normal(0.002, 0.03, len(months)),
            "HML": rng.normal(0.003, 0.03, len(months)),
            "MOM": rng.normal(0.005, 0.04, len(months)),
        }
    )

    print(f"  Fund returns:  {len(fund_returns_df):>8,} obs")
    print(f"  Holdings:      {len(holdings_df):>8,} obs")
    print(f"  Stock returns: {len(stock_returns_df):>8,} obs")
    print(f"  Factors:       {len(factors_df):>8,} obs")

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


# =============================================================================
# SECTION 2: DATA LOADING (REAL DATA MODE)
# =============================================================================

def load_real_data(
    fund_path: str,
    holdings_path: str,
    stock_path: str,
    factor_path: str,
) -> tuple:
    """
    Load pre-processed WRDS data from CSV files.

    Parameters
    ----------
    fund_path : str
        Path to fund_returns.csv.
    holdings_path : str
        Path to holdings.csv (13F filings).
    stock_path : str
        Path to stock_returns.csv.
    factor_path : str
        Path to factors.csv.

    Returns
    -------
    tuple of DataFrames
        (fund_returns_df, holdings_df, stock_returns_df, factors_df)
    """
    print(f"\n{'─'*60}")
    print("  Loading Real Data from CSV Files")
    print(f"{'─'*60}")

    fund_returns_df = pd.read_csv(fund_path, parse_dates=["date"])
    print(f"  Fund returns:  {len(fund_returns_df):>8,} obs  ← {fund_path}")

    holdings_df = pd.read_csv(holdings_path, parse_dates=["report_date"])
    print(f"  Holdings:      {len(holdings_df):>8,} obs  ← {holdings_path}")

    stock_returns_df = pd.read_csv(stock_path, parse_dates=["date"])
    print(f"  Stock returns: {len(stock_returns_df):>8,} obs  ← {stock_path}")

    factors_df = pd.read_csv(factor_path, parse_dates=["date"])
    print(f"  Factors:       {len(factors_df):>8,} obs  ← {factor_path}")

    # Basic validation
    required = {
        "fund_returns": ["date", "fund_id", "fund_ret"],
        "holdings": ["report_date", "fund_id", "stock", "weight"],
        "stock_returns": ["date", "stock", "ret"],
        "factors": ["date"] + FACTOR_COLS,
    }
    dataframes = {
        "fund_returns": fund_returns_df,
        "holdings": holdings_df,
        "stock_returns": stock_returns_df,
        "factors": factors_df,
    }
    for name, cols in required.items():
        missing = [c for c in cols if c not in dataframes[name].columns]
        if missing:
            raise ValueError(f"  ERROR: '{name}' is missing columns: {missing}")

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


# =============================================================================
# SECTION 3: RETURN COMPOUNDING
# =============================================================================

def compound_to_quarterly(df: pd.DataFrame, date_col: str, id_col: str, ret_col: str) -> pd.DataFrame:
    """
    Compound monthly returns to quarterly for a panel of entities.

    Uses geometric compounding: Q_ret = prod(1 + r_monthly) - 1.
    Requires at least MIN_MONTHS_PER_QUARTER months of data per entity-quarter.

    Parameters
    ----------
    df : DataFrame
        Long-format panel with monthly returns.
    date_col : str
        Name of the date column.
    id_col : str
        Name of the entity identifier column.
    ret_col : str
        Name of the return column.

    Returns
    -------
    DataFrame
        Columns: [id_col, 'quarter', ret_col + '_q']
    """
    data = df[[date_col, id_col, ret_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["quarter"] = data[date_col].dt.to_period("Q").dt.to_timestamp("Q")

    # Geometric compounding per entity-quarter
    quarterly = (
        data.groupby([id_col, "quarter"])[ret_col]
        .agg(lambda x: (1 + x).prod() - 1)
        .reset_index()
    )
    quarterly.rename(columns={ret_col: ret_col + "_q"}, inplace=True)

    # Filter by minimum months requirement
    month_count = data.groupby([id_col, "quarter"]).size().reset_index(name="_n")
    quarterly = quarterly.merge(month_count, on=[id_col, "quarter"])
    quarterly = quarterly[quarterly["_n"] >= MIN_MONTHS_PER_QUARTER].drop(columns="_n")

    return quarterly


def compound_monthly_to_quarterly_fund(fund_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly fund returns to quarterly."""
    print("\n  [Step 2a] Compounding fund returns to quarterly...")
    result = compound_to_quarterly(fund_returns_df, "date", "fund_id", "fund_ret")
    n_funds = result["fund_id"].nunique()
    n_quarters = result["quarter"].nunique()
    print(f"           → {len(result):,} obs ({n_funds} funds × {n_quarters} quarters)")
    return result


def compound_monthly_to_quarterly_stock(stock_returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly stock returns to quarterly (wide format)."""
    print("  [Step 2b] Compounding stock returns to quarterly...")
    quarterly = compound_to_quarterly(stock_returns_df, "date", "stock", "ret")
    wide = quarterly.pivot(index="quarter", columns="stock", values="ret_q")
    print(f"           → {wide.shape[0]} quarters × {wide.shape[1]} stocks")
    return wide


def compound_factors_to_quarterly(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly factor returns to quarterly.

    Note: Following standard practice in the literature, Fama-French factors
    are summed (simple addition) within each quarter rather than geometrically
    compounded, since they represent zero-investment portfolio returns where
    simple summation is the convention.
    """
    print("  [Step 2c] Aggregating factors to quarterly...")
    df = factors_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp("Q")

    # Simple sum per quarter (standard for long-short factor portfolios)
    quarterly = df.groupby("quarter")[FACTOR_COLS].sum()
    print(f"           → {len(quarterly)} quarters")
    return quarterly


# =============================================================================
# SECTION 4: UNOBSERVED PERFORMANCE (UP) CONSTRUCTION
# =============================================================================

def _build_holdings_index(holdings_df: pd.DataFrame) -> dict:
    """
    Pre-group holdings by fund_id for O(1) lookup.

    This optimization avoids repeated DataFrame filtering in the inner loop,
    reducing UP computation time from O(N*M) to O(N) where N = fund-quarters
    and M = total holdings rows.
    """
    return {fid: group for fid, group in holdings_df.groupby("fund_id")}


def _buy_and_hold_return(
    fund_id: str,
    quarter: pd.Timestamp,
    holdings_index: dict,
    stock_ret_wide: pd.DataFrame,
) -> float:
    """
    Compute the hypothetical buy-and-hold return for a fund in a given quarter.

    The buy-and-hold return uses the most recent holdings disclosure prior to
    the current quarter, applies those weights to the actual stock returns
    during the current quarter, and computes the weighted portfolio return.

    Parameters
    ----------
    fund_id : str
        Fund identifier.
    quarter : pd.Timestamp
        End-of-quarter timestamp for the return period.
    holdings_index : dict
        Pre-grouped holdings data {fund_id: DataFrame}.
    stock_ret_wide : DataFrame
        Wide-format quarterly stock returns (index=quarter, columns=stocks).

    Returns
    -------
    float or np.nan
        The hypothetical buy-and-hold return, or NaN if unavailable.
    """
    if fund_id not in holdings_index:
        return np.nan

    fund_holdings = holdings_index[fund_id]
    prev_reports = fund_holdings[fund_holdings["report_date"] < quarter]

    if prev_reports.empty:
        return np.nan

    # Use the most recent disclosure prior to this quarter
    prev_quarter = prev_reports["report_date"].max()
    prev_holdings = prev_reports[prev_reports["report_date"] == prev_quarter].copy()

    total_weight = prev_holdings["weight"].sum()
    if total_weight <= 0:
        return np.nan

    # Normalize weights to sum to 1
    prev_holdings["weight"] = prev_holdings["weight"] / total_weight

    if quarter not in stock_ret_wide.index:
        return np.nan

    stock_rets = stock_ret_wide.loc[quarter]

    # Find stocks present in both holdings and return data
    common_stocks = [
        s
        for s in prev_holdings["stock"].values
        if s in stock_rets.index and pd.notna(stock_rets[s])
    ]

    if len(common_stocks) == 0:
        return np.nan

    # Re-normalize weights over available stocks
    weights = prev_holdings.set_index("stock").loc[common_stocks, "weight"]
    weights = weights / weights.sum()
    returns = stock_rets[common_stocks]

    return float((weights * returns).sum())


def calculate_up_panel(
    fund_quarterly: pd.DataFrame,
    holdings_df: pd.DataFrame,
    stock_ret_wide: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute the Unobserved Performance (UP) measure for all fund-quarters.

    UP(i,t) = R_reported(i,t) - R_buyandhold(i,t)

    where R_buyandhold is the return the fund would have earned by simply
    holding its most recently disclosed equity portfolio without trading.

    Parameters
    ----------
    fund_quarterly : DataFrame
        Quarterly fund returns with columns ['fund_id', 'quarter', 'fund_ret_q'].
    holdings_df : DataFrame
        Holdings data with columns ['report_date', 'fund_id', 'stock', 'weight'].
    stock_ret_wide : DataFrame
        Wide-format quarterly stock returns.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    DataFrame
        Panel with columns ['quarter', 'fund_id', 'UP', 'fund_ret_q', 'bh_ret_q'].
    """
    print("\n  [Step 3] Computing Unobserved Performance (UP)...")
    print("           Building holdings index...")

    holdings_index = _build_holdings_index(holdings_df)
    fund_ids = fund_quarterly["fund_id"].unique()
    total = len(fund_ids)
    progress_step = max(1, total // 5)

    results = []

    for idx, fund_id in enumerate(fund_ids):
        if verbose and (idx + 1) % progress_step == 0:
            pct = (idx + 1) / total * 100
            print(f"           Progress: {idx + 1:>5}/{total} funds ({pct:.0f}%)")

        fund_data = fund_quarterly[fund_quarterly["fund_id"] == fund_id]

        for row in fund_data.itertuples(index=False):
            q = row.quarter
            reported_ret = row.fund_ret_q

            bh_ret = _buy_and_hold_return(fund_id, q, holdings_index, stock_ret_wide)

            if pd.isna(bh_ret):
                continue

            results.append(
                {
                    "quarter": q,
                    "fund_id": fund_id,
                    "UP": reported_ret - bh_ret,
                    "fund_ret_q": reported_ret,
                    "bh_ret_q": bh_ret,
                }
            )

    up_panel = pd.DataFrame(results)

    if up_panel.empty:
        print("           ⚠  WARNING: No valid UP observations computed.")
        return up_panel

    n_obs = len(up_panel)
    n_funds = up_panel["fund_id"].nunique()
    n_quarters = up_panel["quarter"].nunique()
    print(f"           → {n_obs:,} observations ({n_funds} funds, {n_quarters} quarters)")
    print(f"           → UP mean = {up_panel['UP'].mean():.5f}, "
          f"std = {up_panel['UP'].std():.5f}, "
          f"median = {up_panel['UP'].median():.5f}")

    return up_panel


# =============================================================================
# SECTION 5: PORTFOLIO SORTS AND LONG-SHORT STRATEGY
# =============================================================================

def form_quintile_portfolios(
    up_panel: pd.DataFrame,
    horizon: int = 3,
    min_funds: int = MIN_FUNDS_PER_PERIOD,
) -> tuple:
    """
    Form quintile portfolios sorted on UP with a specified prediction horizon.

    Each period t:
      1. Sort all funds cross-sectionally by UP(t) into quintiles.
      2. Compute equal-weighted average return in period t + horizon.
      3. Long-Short = Q5 (high UP) minus Q1 (low UP).

    This implements the standard predictive portfolio sort methodology
    used throughout the asset pricing literature.

    Parameters
    ----------
    up_panel : DataFrame
        Panel with columns ['quarter', 'fund_id', 'UP', 'fund_ret_q'].
    horizon : int
        Number of periods ahead for measuring future returns (default=3
        for t+3 months as in the paper's main specification).
    min_funds : int
        Minimum number of funds required per sorting period.

    Returns
    -------
    tuple (ls_df, quintile_df)
        ls_df: DataFrame indexed by period with 'LS_return' column.
        quintile_df: DataFrame with returns for all five quintiles.
    """
    print(f"\n  [Step 4] Forming Quintile Portfolios (horizon = t+{horizon})...")

    quarters = sorted(up_panel["quarter"].unique())
    ls_results = []
    quintile_results = []
    skipped = 0

    for i in range(len(quarters) - horizon):
        sort_q = quarters[i]
        eval_q = quarters[i + horizon]

        # Cross-sectional sort on UP in period t
        sort_data = up_panel[up_panel["quarter"] == sort_q].copy()

        if len(sort_data) < min_funds:
            skipped += 1
            continue

        # Assign quintiles; handle ties gracefully
        try:
            sort_data["quintile"] = pd.qcut(
                sort_data["UP"], q=N_QUINTILES, labels=False, duplicates="drop"
            ) + 1
        except ValueError:
            skipped += 1
            continue

        if sort_data["quintile"].nunique() < N_QUINTILES:
            skipped += 1
            continue

        # Get returns in evaluation period (t + horizon)
        eval_data = up_panel[up_panel["quarter"] == eval_q][["fund_id", "fund_ret_q"]]
        merged = sort_data[["fund_id", "quintile"]].merge(eval_data, on="fund_id", how="inner")

        if len(merged) < min_funds:
            skipped += 1
            continue

        # Equal-weighted portfolio returns per quintile
        port_rets = merged.groupby("quintile")["fund_ret_q"].mean()

        if N_QUINTILES in port_rets.index and 1 in port_rets.index:
            ls_ret = port_rets[N_QUINTILES] - port_rets[1]
            ls_results.append({"quarter": eval_q, "LS_return": ls_ret})

            row = {"quarter": eval_q}
            for q_bin in range(1, N_QUINTILES + 1):
                row[f"Q{q_bin}"] = port_rets.get(q_bin, np.nan)
            row["LS"] = ls_ret
            row["n_funds"] = len(merged)
            quintile_results.append(row)

    ls_df = pd.DataFrame(ls_results)
    if not ls_df.empty:
        ls_df = ls_df.set_index("quarter")

    quintile_df = pd.DataFrame(quintile_results)
    if not quintile_df.empty:
        quintile_df = quintile_df.set_index("quarter")

    n_valid = len(ls_df)
    print(f"           → {n_valid} evaluation periods (skipped {skipped})")

    if n_valid > 0:
        mean_ls = ls_df["LS_return"].mean()
        std_ls = ls_df["LS_return"].std(ddof=1)
        t_simple = mean_ls / (std_ls / np.sqrt(n_valid)) if std_ls > 0 else np.nan
        print(f"           → Mean L/S return: {mean_ls*100:.3f}% per quarter")
        print(f"           → Std:             {std_ls*100:.3f}%")
        print(f"           → t-statistic:     {t_simple:.2f}")

    return ls_df, quintile_df


# =============================================================================
# SECTION 6: RISK-ADJUSTED ALPHA (FACTOR MODEL)
# =============================================================================

def compute_alpha(
    ls_df: pd.DataFrame,
    factors_quarterly: pd.DataFrame,
    max_lags: int = NEWEY_WEST_LAGS,
) -> dict:
    """
    Estimate risk-adjusted alpha from a Carhart (1997) four-factor model.

    Model:
        LS_t = alpha + beta_MKT * MKT_t + beta_SMB * SMB_t
                     + beta_HML * HML_t + beta_MOM * MOM_t + epsilon_t

    Standard errors are computed using Newey-West (1987) HAC estimator
    to account for potential serial correlation and heteroskedasticity.

    Parameters
    ----------
    ls_df : DataFrame
        Long-short portfolio returns, indexed by quarter.
    factors_quarterly : DataFrame
        Quarterly factor returns, indexed by quarter.
    max_lags : int
        Maximum lag order for Newey-West estimator.

    Returns
    -------
    dict
        Regression results including alpha, t-statistic, and full model.
    """
    print(f"\n  [Step 5] Carhart 4-Factor Alpha (Newey-West, {max_lags} lags)")
    print(f"  {'─'*56}")

    combined = ls_df[["LS_return"]].join(factors_quarterly, how="inner").dropna()

    if len(combined) < 12:
        print("           ⚠  Insufficient observations (need ≥ 12).")
        return None

    y = combined["LS_return"]
    X = sm.add_constant(combined[FACTOR_COLS])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max_lags})

    # Display results
    print(f"  {'Variable':<10} {'Coef':>10} {'Std Err':>10} {'t-stat':>8} {'p-val':>8}")
    print(f"  {'─'*48}")
    for var in model.params.index:
        label = "Alpha" if var == "const" else var
        coef = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {label:<10} {coef:>10.5f} {se:>10.5f} {t:>8.2f} {p:>8.4f} {sig}")
    print(f"  {'─'*48}")
    print(f"  R²  = {model.rsquared:.4f}    N = {int(model.nobs)}")

    alpha_q = model.params["const"]
    alpha_ann = alpha_q * 4
    t_stat = model.tvalues["const"]

    print(f"\n  Key Result:")
    print(f"    Quarterly alpha = {alpha_q:.5f} ({alpha_q*100:.3f}%)")
    print(f"    Annualized alpha = {alpha_ann:.5f} ({alpha_ann*100:.2f}%)")
    print(f"    t-statistic = {t_stat:.3f}", end="")
    if abs(t_stat) > 2.576:
        print(" [significant at 1% level]")
    elif abs(t_stat) > 1.96:
        print(" [significant at 5% level]")
    elif abs(t_stat) > 1.645:
        print(" [significant at 10% level]")
    else:
        print(" [not statistically significant]")

    return {
        "model": model,
        "alpha_quarterly": alpha_q,
        "alpha_annualized": alpha_ann,
        "t_stat": t_stat,
        "p_value": model.pvalues["const"],
        "n_obs": int(model.nobs),
        "r_squared": model.rsquared,
    }


# =============================================================================
# SECTION 7: QUINTILE MONOTONICITY ANALYSIS
# =============================================================================

def analyze_quintile_returns(
    quintile_df: pd.DataFrame,
    factors_quarterly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze whether portfolio returns increase monotonically from Q1 to Q5.

    This is a key test of the paper's hypothesis: if UP captures managerial
    skill, then high-UP funds should outperform low-UP funds in the future.

    Parameters
    ----------
    quintile_df : DataFrame
        Quintile portfolio returns (columns Q1...Q5, LS, n_funds).
    factors_quarterly : DataFrame
        Quarterly factor returns for alpha estimation.

    Returns
    -------
    DataFrame
        Summary statistics and alphas for each quintile.
    """
    print(f"\n  [Step 6] Quintile Portfolio Analysis")
    print(f"  {'═'*65}")
    print(f"  {'Portfolio':<12} {'Mean%':>8} {'Ann%':>8} {'Std%':>8} {'Alpha%':>8} {'t(α)':>7}")
    print(f"  {'─'*65}")

    summary_rows = []

    for q_bin in range(1, N_QUINTILES + 1):
        col = f"Q{q_bin}"
        if col not in quintile_df.columns:
            continue

        rets = quintile_df[col].dropna()
        mean_q = rets.mean()
        std_q = rets.std(ddof=1)
        ann_ret = mean_q * 4

        # Compute alpha
        alpha, t_alpha = np.nan, np.nan
        combined = rets.to_frame("ret").join(factors_quarterly, how="inner").dropna()
        if len(combined) >= 12:
            y = combined["ret"]
            X = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS})
            alpha = mod.params["const"]
            t_alpha = mod.tvalues["const"]

        label = f"Q{q_bin} (Low)" if q_bin == 1 else f"Q{q_bin} (High)" if q_bin == N_QUINTILES else f"Q{q_bin}"
        print(
            f"  {label:<12} {mean_q*100:>8.3f} {ann_ret*100:>8.2f} "
            f"{std_q*100:>8.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )
        summary_rows.append(
            {
                "quintile": q_bin,
                "mean_quarterly": mean_q,
                "annualized": ann_ret,
                "std": std_q,
                "alpha": alpha,
                "t_alpha": t_alpha,
            }
        )

    # Long-Short
    if "LS" in quintile_df.columns:
        rets = quintile_df["LS"].dropna()
        mean_q = rets.mean()
        std_q = rets.std(ddof=1)
        combined = rets.to_frame("ret").join(factors_quarterly, how="inner").dropna()
        alpha, t_alpha = np.nan, np.nan
        if len(combined) >= 12:
            y = combined["ret"]
            X = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NEWEY_WEST_LAGS})
            alpha = mod.params["const"]
            t_alpha = mod.tvalues["const"]
        print(f"  {'─'*65}")
        print(
            f"  {'Q5-Q1':<12} {mean_q*100:>8.3f} {mean_q*400:>8.2f} "
            f"{std_q*100:>8.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )

    print(f"  {'═'*65}")
    return pd.DataFrame(summary_rows)


# =============================================================================
# SECTION 8: UP PERSISTENCE ANALYSIS
# =============================================================================

def analyze_persistence(up_panel: pd.DataFrame) -> dict:
    """
    Test whether UP exhibits time-series persistence using Fama-MacBeth regressions.

    If UP reflects persistent managerial skill, we expect significant positive
    autocorrelation: UP(t) should predict UP(t+1) cross-sectionally.

    The test runs quarterly cross-sectional regressions of UP(t) on UP(t-1),
    then computes the time-series average coefficient and its t-statistic.

    Parameters
    ----------
    up_panel : DataFrame
        UP panel data.

    Returns
    -------
    dict
        Persistence coefficient and t-statistic.
    """
    print(f"\n  [Step 7] UP Persistence Analysis (Fama-MacBeth)")
    print(f"  {'─'*50}")

    panel = up_panel[["quarter", "fund_id", "UP"]].copy()
    panel = panel.sort_values(["fund_id", "quarter"])
    panel["UP_lag1"] = panel.groupby("fund_id")["UP"].shift(1)

    valid = panel.dropna(subset=["UP", "UP_lag1"])

    if len(valid) < 50:
        print("           ⚠  Insufficient data for persistence test.")
        return {}

    # Fama-MacBeth: cross-sectional regression each quarter
    quarters = sorted(valid["quarter"].unique())
    coeffs = []

    for q in quarters:
        q_data = valid[valid["quarter"] == q]
        if len(q_data) < 10:
            continue
        X = sm.add_constant(q_data["UP_lag1"])
        y = q_data["UP"]
        try:
            model = sm.OLS(y, X).fit()
            coeffs.append(model.params["UP_lag1"])
        except Exception:
            continue

    if not coeffs:
        print("           ⚠  Could not estimate persistence.")
        return {}

    mean_coeff = np.mean(coeffs)
    se = np.std(coeffs, ddof=1) / np.sqrt(len(coeffs))
    t_stat = mean_coeff / se if se > 0 else np.nan

    print(f"  Fama-MacBeth AR(1) coefficient: {mean_coeff:.4f}")
    print(f"  Standard error:                 {se:.4f}")
    print(f"  t-statistic:                    {t_stat:.2f}")
    print(f"  Number of cross-sections:       {len(coeffs)}")

    if abs(t_stat) > 1.96:
        print("  → UP exhibits statistically significant persistence (p < 0.05)")
    else:
        print("  → UP persistence is not statistically significant")

    return {"ar1_coefficient": mean_coeff, "t_stat": t_stat, "n_periods": len(coeffs)}


# =============================================================================
# SECTION 9: VALIDATION AGAINST PUBLISHED RESULTS
# =============================================================================

def validate_against_paper(
    csv_path: str,
    quintile_df: pd.DataFrame = None,
) -> None:
    """
    Load and summarize the published portfolio sort results for comparison.

    The validation file contains monthly returns for quintile portfolios sorted
    on UP(t) with evaluation in t+3, as reported in the paper.

    Parameters
    ----------
    csv_path : str
        Path to the published results CSV file.
    quintile_df : DataFrame, optional
        Replicated quintile returns for direct comparison.
    """
    print(f"\n  {'═'*65}")
    print(f"  VALIDATION: Published Results from Weigert et al. (JF 2024)")
    print(f"  {'═'*65}")

    df = pd.read_csv(csv_path, skiprows=1)

    # Standardize column names
    df.columns = [
        "Year", "Month", "PF1", "PF2", "PF3", "PF4", "PF5", "LS"
    ]

    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)

    # Convert from percentage to decimal
    for col in ["PF1", "PF2", "PF3", "PF4", "PF5", "LS"]:
        df[col] = df[col] / 100.0

    n_months = len(df)
    date_range = f"{df['Year'].min()}-{df['Month'].iloc[0]:02d} to {df['Year'].max()}-{df['Month'].iloc[-1]:02d}"

    print(f"\n  Sample: {date_range} ({n_months} months)")
    print(f"\n  {'Portfolio':<12} {'Mean%/mo':>10} {'Ann%':>8} {'Std%/mo':>10} {'t-stat':>8}")
    print(f"  {'─'*50}")

    for col, label in [
        ("PF1", "Q1 (Low UP)"),
        ("PF2", "Q2"),
        ("PF3", "Q3"),
        ("PF4", "Q4"),
        ("PF5", "Q5 (High UP)"),
        ("LS", "Q5 - Q1"),
    ]:
        mean_m = df[col].mean()
        std_m = df[col].std(ddof=1)
        t = mean_m / (std_m / np.sqrt(n_months)) if std_m > 0 else np.nan
        ann = mean_m * 12
        if col == "LS":
            print(f"  {'─'*50}")
        print(f"  {label:<12} {mean_m*100:>10.4f} {ann*100:>8.2f} {std_m*100:>10.4f} {t:>8.2f}")

    print(f"  {'═'*65}")

    # Summary interpretation
    ls_mean = df["LS"].mean()
    ls_std = df["LS"].std(ddof=1)
    ls_t = ls_mean / (ls_std / np.sqrt(n_months))

    print(f"\n  Published Long-Short Strategy Summary:")
    print(f"    Monthly mean return: {ls_mean*100:.4f}%")
    print(f"    Annualized return:   {ls_mean*1200:.2f}%")
    print(f"    Monthly volatility:  {ls_std*100:.4f}%")
    print(f"    t-statistic:         {ls_t:.3f}")

    if abs(ls_t) > 1.96:
        print(f"    → Statistically significant at 5% level")
    else:
        print(f"    → Not statistically significant at 5% level")

    return df


# =============================================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replicate Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replicate_up.py --demo
  python replicate_up.py --fund-returns data/fund_ret.csv --holdings data/holdings.csv \\
                         --stock-returns data/stock_ret.csv --factors data/factors.csv
  python replicate_up.py --demo --validate data/UP_5-1.csv --horizon 3
        """,
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (no external files needed).",
    )
    parser.add_argument("--fund-returns", type=str, help="Path to fund_returns.csv.")
    parser.add_argument("--holdings", type=str, help="Path to holdings.csv.")
    parser.add_argument("--stock-returns", type=str, help="Path to stock_returns.csv.")
    parser.add_argument("--factors", type=str, help="Path to factors.csv.")
    parser.add_argument(
        "--validate", type=str, default=None,
        help="Path to published results CSV for validation.",
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Prediction horizon in quarters (default=1; paper uses monthly with t+3).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/results",
        help="Directory for output files (default: data/results).",
    )
    parser.add_argument(
        "--n-funds", type=int, default=100,
        help="Number of funds in demo mode (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()

    print("=" * 70)
    print("  Replicating: Unobserved Performance of Hedge Funds")
    print("  Weigert, Wegener, & Klesczewski (Journal of Finance, 2024)")
    print(f"  Pipeline version: {__version__}")
    print("=" * 70)

    # ─── Step 1: Load Data ─────────────────────────────────────────────────
    if args.demo:
        fund_returns_df, holdings_df, stock_returns_df, factors_df = generate_demo_data(
            n_funds=args.n_funds, seed=args.seed
        )
    else:
        if not all([args.fund_returns, args.holdings, args.stock_returns, args.factors]):
            print("\n  ERROR: Must provide all four data files or use --demo.")
            print("  Run with --help for usage information.")
            sys.exit(1)
        fund_returns_df, holdings_df, stock_returns_df, factors_df = load_real_data(
            args.fund_returns, args.holdings, args.stock_returns, args.factors
        )

    # ─── Step 2: Compound to Quarterly ─────────────────────────────────────
    fund_quarterly = compound_monthly_to_quarterly_fund(fund_returns_df)
    stock_ret_wide = compound_monthly_to_quarterly_stock(stock_returns_df)
    factors_quarterly = compound_factors_to_quarterly(factors_df)

    # ─── Step 3: Calculate UP Panel ────────────────────────────────────────
    up_panel = calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide)

    if up_panel.empty:
        print("\n  FATAL: No valid UP observations. Check data alignment.")
        sys.exit(1)

    # ─── Step 4: Portfolio Sorts ───────────────────────────────────────────
    ls_df, quintile_df = form_quintile_portfolios(
        up_panel, horizon=args.horizon, min_funds=MIN_FUNDS_PER_PERIOD
    )

    if ls_df.empty:
        print("\n  WARNING: Could not form long-short portfolio.")
    else:
        # ─── Step 5: Factor Model Alpha ────────────────────────────────────
        alpha_results = compute_alpha(ls_df, factors_quarterly)

        # ─── Step 6: Quintile Analysis ─────────────────────────────────────
        if not quintile_df.empty:
            analyze_quintile_returns(quintile_df, factors_quarterly)

    # ─── Step 7: Persistence ───────────────────────────────────────────────
    analyze_persistence(up_panel)

    # ─── Step 8: Validate Against Published Results ────────────────────────
    if args.validate:
        validate_path = Path(args.validate)
        if validate_path.exists():
            validate_against_paper(str(validate_path), quintile_df)
        else:
            print(f"\n  WARNING: Validation file not found: {args.validate}")

    # ─── Step 9: Save Results ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    up_panel.to_csv(output_dir / "up_panel.csv", index=False)
    print(f"\n  Saved: {output_dir / 'up_panel.csv'}")

    if not ls_df.empty:
        ls_df.to_csv(output_dir / "long_short_returns.csv")
        print(f"  Saved: {output_dir / 'long_short_returns.csv'}")

    if not quintile_df.empty:
        quintile_df.to_csv(output_dir / "quintile_returns.csv")
        print(f"  Saved: {output_dir / 'quintile_returns.csv'}")

    # ─── Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Replication Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
