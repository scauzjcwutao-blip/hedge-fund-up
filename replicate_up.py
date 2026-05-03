"""
Unobserved Performance of Hedge Funds (Weigert et al., Journal of Finance 2024)
Python Replication Script - Final Corrected Full Version
Author: Tao Wu
Date: May 2026

## Key Features
- Full multi-fund panel processing with correct cross-sectional quintile sorting
- Accurate monthly-to-quarterly compounding
- Predictive long-short portfolios (t → t+1 or t+3)
- Carhart 4-factor alphas with Newey-West HAC standard errors
- Built-in validation against the paper's official UP 5-1 benchmark
- Clean command-line interface (demo + real WRDS data)
"""

import argparse
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

__version__ = "2.0.0"

np.random.seed(42)

print("=" * 80)
print("Replicating Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds")
print("Final Corrected Version: Multi-Fund Panel + Predictive Sort")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_FUNDS_PER_QUARTER = 20
N_QUINTILES = 5
NEWEY_WEST_LAGS = 4
FACTOR_COLS = ['MKT', 'SMB', 'HML', 'MOM']


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def generate_demo_data(n_funds=100, n_stocks=50):
    """Generate realistic demo data for testing."""
    print(f"\n{'─'*60}")
    print(f"  Generating Demo Data: {n_funds} funds, {n_stocks} stocks")
    print(f"  Period: 1994-01 to 2019-12")
    print(f"{'─'*60}")

    months = pd.date_range(start='1994-01-31', end='2019-12-31', freq='ME')
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
    fund_ids = [f"FUND{i:04d}" for i in range(1, n_funds + 1)]
    stock_ids = [f"STK{i:03d}" for i in range(1, n_stocks + 1)]

    # Fund monthly returns with heterogeneous skill
    fund_records = []
    for fid in fund_ids:
        alpha = np.random.normal(0.005, 0.003)
        vol = np.random.uniform(0.02, 0.06)
        rets = np.random.normal(alpha, vol, len(months))
        for date, ret in zip(months, rets):
            fund_records.append({'date': date, 'fund_id': fid, 'fund_ret': ret})
    fund_returns_df = pd.DataFrame(fund_records)

    # Stock monthly returns
    stock_records = []
    for sid in stock_ids:
        mu = np.random.normal(0.008, 0.005)
        vol = np.random.uniform(0.04, 0.12)
        rets = np.random.normal(mu, vol, len(months))
        for date, ret in zip(months, rets):
            stock_records.append({'date': date, 'stock': sid, 'ret': ret})
    stock_returns_df = pd.DataFrame(stock_records)

    # Quarterly holdings per fund
    holdings_records = []
    for fid in fund_ids:
        n_hold = np.random.randint(5, 20)
        current_stocks = np.random.choice(stock_ids, n_hold, replace=False)
        for q in quarters:
            if np.random.random() < 0.3:
                n_hold = np.random.randint(5, 20)
                current_stocks = np.random.choice(stock_ids, n_hold, replace=False)
            weights = np.random.dirichlet(np.ones(len(current_stocks)))
            for stock, weight in zip(current_stocks, weights):
                holdings_records.append({
                    'report_date': q, 'fund_id': fid,
                    'stock': stock, 'weight': weight
                })
    holdings_df = pd.DataFrame(holdings_records)

    # Monthly factors
    factors_df = pd.DataFrame({
        'date': months,
        'MKT': np.random.normal(0.006, 0.04, len(months)),
        'SMB': np.random.normal(0.002, 0.03, len(months)),
        'HML': np.random.normal(0.003, 0.03, len(months)),
        'MOM': np.random.normal(0.005, 0.04, len(months)),
    })

    print(f"  Fund returns:  {len(fund_returns_df):>8,} obs")
    print(f"  Holdings:      {len(holdings_df):>8,} obs")
    print(f"  Stock returns: {len(stock_returns_df):>8,} obs")
    print(f"  Factors:       {len(factors_df):>8,} obs")

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


def load_real_data(fund_path, holdings_path, stock_path, factor_path):
    """Load real WRDS CSV data from specified paths."""
    print(f"\n{'─'*60}")
    print("  Loading Real Data from CSV Files")
    print(f"{'─'*60}")

    fund_returns_df = pd.read_csv(fund_path, parse_dates=['date'])
    print(f"  Fund returns:  {len(fund_returns_df):>8,} obs  ← {fund_path}")

    holdings_df = pd.read_csv(holdings_path, parse_dates=['report_date'])
    print(f"  Holdings:      {len(holdings_df):>8,} obs  ← {holdings_path}")

    stock_returns_df = pd.read_csv(stock_path, parse_dates=['date'])
    print(f"  Stock returns: {len(stock_returns_df):>8,} obs  ← {stock_path}")

    factors_df = pd.read_csv(factor_path, parse_dates=['date'])
    print(f"  Factors:       {len(factors_df):>8,} obs  ← {factor_path}")

    # Validate required columns
    required = {
        'fund_returns': (['date', 'fund_id', 'fund_ret'], fund_returns_df),
        'holdings': (['report_date', 'fund_id', 'stock', 'weight'], holdings_df),
        'stock_returns': (['date', 'stock', 'ret'], stock_returns_df),
        'factors': (['date'] + FACTOR_COLS, factors_df),
    }
    for name, (cols, df) in required.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"'{name}' is missing columns: {missing}")

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


# =============================================================================
# 2. COMPOUND MONTHLY TO QUARTERLY
# =============================================================================

def compound_monthly_to_quarterly(fund_returns_df):
    """Compound monthly fund returns into quarterly returns."""
    print("\n  [Step 2a] Compounding fund returns to quarterly...")
    df = fund_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby(['fund_id', 'quarter']).apply(
        lambda x: (1 + x['fund_ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['fund_id', 'quarter', 'fund_ret_q']

    month_count = df.groupby(['fund_id', 'quarter']).size().reset_index(name='n_months')
    quarterly = quarterly.merge(month_count, on=['fund_id', 'quarter'])
    quarterly = quarterly[quarterly['n_months'] >= 2].drop(columns='n_months')

    n_funds = quarterly['fund_id'].nunique()
    n_quarters = quarterly['quarter'].nunique()
    print(f"           → {len(quarterly):,} obs ({n_funds} funds × {n_quarters} quarters)")
    return quarterly


def compound_stock_returns_quarterly(stock_returns_df):
    """Compound monthly stock returns to quarterly (wide format)."""
    print("  [Step 2b] Compounding stock returns to quarterly...")
    df = stock_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby(['stock', 'quarter']).apply(
        lambda x: (1 + x['ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['stock', 'quarter', 'ret_q']

    stock_ret_wide = quarterly.pivot(index='quarter', columns='stock', values='ret_q')
    print(f"           → {stock_ret_wide.shape[0]} quarters × {stock_ret_wide.shape[1]} stocks")
    return stock_ret_wide


def compound_factors_quarterly(factors_df):
    """Compound monthly factors to quarterly."""
    print("  [Step 2c] Compounding factors to quarterly...")
    df = factors_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby('quarter')[FACTOR_COLS].apply(
        lambda x: (1 + x).prod() - 1
    )
    print(f"           → {len(quarterly)} quarters")
    return quarterly


# =============================================================================
# 3. OPTIMIZED UP PANEL CALCULATION
# =============================================================================

def build_holdings_index(holdings_df):
    """Pre-group holdings by fund_id for O(1) lookup."""
    return {fid: group for fid, group in holdings_df.groupby('fund_id')}


def construct_buy_and_hold_return(fund_id, quarter, holdings_by_fund, stock_ret_wide):
    """
    Compute hypothetical buy-and-hold return from most recent 13F disclosure.

    Uses the latest holdings report *before* the current quarter, applies those
    weights to actual stock returns in the current quarter.
    """
    if fund_id not in holdings_by_fund:
        return np.nan

    fund_holdings = holdings_by_fund[fund_id]
    prev_reports = fund_holdings[fund_holdings['report_date'] < quarter]

    if prev_reports.empty:
        return np.nan

    prev_quarter = prev_reports['report_date'].max()
    prev_holdings = prev_reports[prev_reports['report_date'] == prev_quarter].copy()

    total_weight = prev_holdings['weight'].sum()
    if total_weight <= 0:
        return np.nan
    prev_holdings['weight'] = prev_holdings['weight'] / total_weight

    if quarter not in stock_ret_wide.index:
        return np.nan

    stock_rets = stock_ret_wide.loc[quarter]
    common_stocks = [s for s in prev_holdings['stock'].values
                     if s in stock_rets.index and not pd.isna(stock_rets[s])]

    if len(common_stocks) == 0:
        return np.nan

    weights = prev_holdings.set_index('stock').loc[common_stocks, 'weight']
    weights = weights / weights.sum()
    returns = stock_rets[common_stocks]

    return float((weights * returns).sum())


def calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide):
    """
    Calculate UP panel for all funds.

    UP(i,t) = R_reported(i,t) - R_buyandhold(i,t)
    """
    print("\n  [Step 3] Computing Unobserved Performance (UP)...")
    print("           Building holdings index...")
    holdings_by_fund = build_holdings_index(holdings_df)

    fund_ids = fund_quarterly['fund_id'].unique()
    total = len(fund_ids)
    progress_step = max(1, total // 5)

    results = []

    for idx, fund_id in enumerate(fund_ids):
        if (idx + 1) % progress_step == 0:
            pct = (idx + 1) / total * 100
            print(f"           Progress: {idx + 1:>5}/{total} funds ({pct:.0f}%)")

        fund_data = fund_quarterly[fund_quarterly['fund_id'] == fund_id]
        for _, row in fund_data.iterrows():
            q = row['quarter']
            reported_ret = row['fund_ret_q']
            bh_ret = construct_buy_and_hold_return(
                fund_id, q, holdings_by_fund, stock_ret_wide
            )

            if pd.isna(bh_ret):
                continue

            results.append({
                'quarter': q,
                'fund_id': fund_id,
                'UP': reported_ret - bh_ret,
                'fund_ret_q': reported_ret,
                'bh_ret_q': bh_ret,
            })

    up_panel = pd.DataFrame(results)

    if up_panel.empty:
        print("           ⚠  WARNING: No valid UP observations computed.")
        return up_panel

    n_obs = len(up_panel)
    n_funds = up_panel['fund_id'].nunique()
    n_quarters = up_panel['quarter'].nunique()
    print(f"           → {n_obs:,} observations ({n_funds} funds, {n_quarters} quarters)")
    print(f"           → UP mean = {up_panel['UP'].mean():.5f}, "
          f"std = {up_panel['UP'].std():.5f}, "
          f"median = {up_panel['UP'].median():.5f}")

    return up_panel


# =============================================================================
# 4. PREDICTIVE LONG-SHORT PORTFOLIO (returns both ls_df AND quintile_df)
# =============================================================================

def form_long_short_portfolio(up_panel, horizon=1, min_funds_per_quarter=20):
    """
    Cross-sectional quintile sort on UP(t) → t+horizon actual fund returns.

    Returns
    -------
    tuple (ls_df, quintile_df)
        ls_df: DataFrame indexed by quarter with 'LS_return' column.
        quintile_df: DataFrame with Q1-Q5 and LS returns per evaluation quarter.
    """
    print(f"\n  [Step 4] Forming Long-Short Portfolio (horizon = t+{horizon})...")
    quarters = sorted(up_panel['quarter'].unique())
    ls_results = []
    quintile_results = []
    skipped = 0

    for i in range(len(quarters) - horizon):
        sort_q = quarters[i]
        eval_q = quarters[i + horizon]

        # Cross-sectional sort on UP in period t
        sort_data = up_panel[up_panel['quarter'] == sort_q].copy()

        if len(sort_data) < min_funds_per_quarter:
            skipped += 1
            continue

        # Assign quintiles
        try:
            sort_data['quintile'] = pd.qcut(
                sort_data['UP'], q=N_QUINTILES, labels=False, duplicates='drop'
            ) + 1
        except ValueError:
            skipped += 1
            continue

        if sort_data['quintile'].nunique() < N_QUINTILES:
            skipped += 1
            continue

        # Get actual fund returns in evaluation period (t + horizon)
        eval_data = up_panel[up_panel['quarter'] == eval_q][['fund_id', 'fund_ret_q']]
        merged = sort_data[['fund_id', 'quintile']].merge(
            eval_data, on='fund_id', how='inner'
        )

        if len(merged) < min_funds_per_quarter:
            skipped += 1
            continue

        # Equal-weighted portfolio returns per quintile
        port_rets = merged.groupby('quintile')['fund_ret_q'].mean()

        if N_QUINTILES in port_rets.index and 1 in port_rets.index:
            ls_ret = port_rets[N_QUINTILES] - port_rets[1]
            ls_results.append({'quarter': eval_q, 'LS_return': ls_ret})

            row = {'quarter': eval_q}
            for q_bin in range(1, N_QUINTILES + 1):
                row[f'Q{q_bin}'] = port_rets.get(q_bin, np.nan)
            row['LS'] = ls_ret
            row['n_funds'] = len(merged)
            quintile_results.append(row)

    ls_df = pd.DataFrame(ls_results)
    if not ls_df.empty:
        ls_df = ls_df.set_index('quarter')

    quintile_df = pd.DataFrame(quintile_results)
    if not quintile_df.empty:
        quintile_df = quintile_df.set_index('quarter')

    n_valid = len(ls_df)
    print(f"           → {n_valid} evaluation periods (skipped {skipped})")

    if n_valid > 0:
        mean_ls = ls_df['LS_return'].mean()
        std_ls = ls_df['LS_return'].std(ddof=1)
        t_simple = mean_ls / (std_ls / np.sqrt(n_valid)) if std_ls > 0 else np.nan
        print(f"           → Mean L/S return: {mean_ls*100:.3f}% per quarter")
        print(f"           → Std:             {std_ls*100:.3f}%")
        print(f"           → t-statistic:     {t_simple:.2f}")

    return ls_df, quintile_df


# =============================================================================
# 5. RISK-ADJUSTED ALPHA (Carhart 4-Factor, Newey-West)
# =============================================================================

def compute_risk_adjusted_alpha(ls_df, factors_quarterly, max_lags=NEWEY_WEST_LAGS):
    """
    Carhart 4-factor regression with Newey-West standard errors.

    Model: LS_t = alpha + b1*MKT_t + b2*SMB_t + b3*HML_t + b4*MOM_t + eps_t
    """
    print(f"\n  [Step 5] Carhart 4-Factor Alpha (Newey-West, {max_lags} lags)")
    print(f"  {'─'*56}")

    combined = ls_df[['LS_return']].join(factors_quarterly, how='inner').dropna()

    if len(combined) < 12:
        print("           ⚠  Insufficient observations (need ≥ 12).")
        return None

    y = combined['LS_return']
    X = sm.add_constant(combined[FACTOR_COLS])

    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

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
    print(f"  R² = {model.rsquared:.4f}    N = {int(model.nobs)}")

    alpha_q = model.params['const']
    alpha_ann = alpha_q * 4
    t_stat = model.tvalues['const']

    print(f"\n  Key Result:")
    print(f"    Quarterly alpha  = {alpha_q*100:.3f}%")
    print(f"    Annualized alpha = {alpha_ann*100:.2f}%")
    print(f"    t-statistic      = {t_stat:.3f}", end="")
    if abs(t_stat) > 2.576:
        print(" [significant at 1% level]")
    elif abs(t_stat) > 1.96:
        print(" [significant at 5% level]")
    elif abs(t_stat) > 1.645:
        print(" [significant at 10% level]")
    else:
        print(" [not statistically significant]")

    return {
        'model': model,
        'alpha_quarterly': alpha_q,
        'alpha_annualized': alpha_ann,
        't_stat': t_stat,
        'p_value': model.pvalues['const'],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
    }


# =============================================================================
# 6. QUINTILE MONOTONICITY ANALYSIS
# =============================================================================

def analyze_quintile_returns(quintile_df, factors_quarterly):
    """
    Analyze whether portfolio returns increase monotonically from Q1 to Q5.
    Estimates individual alpha for each quintile portfolio.
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

        # Compute alpha for each quintile
        alpha, t_alpha = np.nan, np.nan
        combined = rets.to_frame('ret').join(factors_quarterly, how='inner').dropna()
        if len(combined) >= 12:
            y_q = combined['ret']
            X_q = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y_q, X_q).fit(
                cov_type='HAC', cov_kwds={'maxlags': NEWEY_WEST_LAGS}
            )
            alpha = mod.params['const']
            t_alpha = mod.tvalues['const']

        label = (f"Q{q_bin} (Low)" if q_bin == 1
                 else f"Q{q_bin} (High)" if q_bin == N_QUINTILES
                 else f"Q{q_bin}")
        print(
            f"  {label:<12} {mean_q*100:>8.3f} {ann_ret*100:>8.2f} "
            f"{std_q*100:>8.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )
        summary_rows.append({
            'quintile': q_bin,
            'mean_quarterly': mean_q,
            'annualized': ann_ret,
            'std': std_q,
            'alpha': alpha,
            't_alpha': t_alpha,
        })

    # Long-Short row
    if 'LS' in quintile_df.columns:
        rets = quintile_df['LS'].dropna()
        mean_q = rets.mean()
        std_q = rets.std(ddof=1)
        combined = rets.to_frame('ret').join(factors_quarterly, how='inner').dropna()
        alpha, t_alpha = np.nan, np.nan
        if len(combined) >= 12:
            y_q = combined['ret']
            X_q = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y_q, X_q).fit(
                cov_type='HAC', cov_kwds={'maxlags': NEWEY_WEST_LAGS}
            )
            alpha = mod.params['const']
            t_alpha = mod.tvalues['const']
        print(f"  {'─'*65}")
        print(
            f"  {'Q5-Q1':<12} {mean_q*100:>8.3f} {mean_q*400:>8.2f} "
            f"{std_q*100:>8.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )

    print(f"  {'═'*65}")
    return pd.DataFrame(summary_rows)


# =============================================================================
# 7. UP PERSISTENCE ANALYSIS (Fama-MacBeth)
# =============================================================================

def analyze_persistence(up_panel):
    """
    Test UP persistence using Fama-MacBeth cross-sectional regressions.
    If UP reflects persistent skill, UP(t-1) should predict UP(t).
    """
    print(f"\n  [Step 7] UP Persistence Analysis (Fama-MacBeth)")
    print(f"  {'─'*50}")

    panel = up_panel[['quarter', 'fund_id', 'UP']].copy()
    panel = panel.sort_values(['fund_id', 'quarter'])
    panel['UP_lag1'] = panel.groupby('fund_id')['UP'].shift(1)

    valid = panel.dropna(subset=['UP', 'UP_lag1'])

    if len(valid) < 50:
        print("           ⚠  Insufficient data for persistence test.")
        return {}

    # Fama-MacBeth: cross-sectional regression each quarter
    quarters = sorted(valid['quarter'].unique())
    coeffs = []

    for q in quarters:
        q_data = valid[valid['quarter'] == q]
        if len(q_data) < 10:
            continue
        X = sm.add_constant(q_data['UP_lag1'])
        y = q_data['UP']
        try:
            model = sm.OLS(y, X).fit()
            coeffs.append(model.params['UP_lag1'])
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

    return {'ar1_coefficient': mean_coeff, 't_stat': t_stat, 'n_periods': len(coeffs)}


# =============================================================================
# 8. VALIDATION AGAINST PUBLISHED RESULTS
# =============================================================================

def validate_against_paper(csv_path, quintile_df=None):
    """
    Load published portfolio sort results and compare with replication.

    The validation CSV contains monthly returns for quintile portfolios
    sorted on UP(t) with evaluation in t+3, as reported in the paper.
    """
    print(f"\n  {'═'*65}")
    print(f"  VALIDATION: Published Results from Weigert et al. (JF 2024)")
    print(f"  {'═'*65}")

    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = ["Year", "Month", "PF1", "PF2", "PF3", "PF4", "PF5", "LS"]
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)

    # Convert from percentage to decimal
    for col in ["PF1", "PF2", "PF3", "PF4", "PF5", "LS"]:
        df[col] = df[col] / 100.0

    n_months = len(df)
    date_range = (f"{df['Year'].min()}-{df['Month'].iloc[0]:02d} to "
                  f"{df['Year'].max()}-{df['Month'].iloc[-1]:02d}")

    print(f"\n  Sample: {date_range} ({n_months} months)")
    print(f"\n  {'Portfolio':<12} {'Mean%/mo':>10} {'Ann%':>8} "
          f"{'Std%/mo':>10} {'t-stat':>8}")
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
        print(f"  {label:<12} {mean_m*100:>10.4f} {ann*100:>8.2f} "
              f"{std_m*100:>10.4f} {t:>8.2f}")

    print(f"  {'═'*65}")

    # Summary
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

    # Direct comparison with replicated results
    if quintile_df is not None and not quintile_df.empty and 'LS' in quintile_df.columns:
        print(f"\n  {'─'*70}")
        print(f"  Comparison: Replicated (quarterly) vs. Published (monthly × 3)")
        print(f"  {'Portfolio':<10} {'Replicated%/qtr':>16} {'Pub×3 %/qtr':>14} {'Diff':>10}")
        print(f"  {'─'*52}")

        for q in range(1, 6):
            col_rep = f"Q{q}"
            col_pub = f"PF{q}"
            if col_rep in quintile_df.columns:
                rep_mean = quintile_df[col_rep].mean() * 100
                pub_approx = df[col_pub].mean() * 300  # monthly → quarterly approx
                diff = rep_mean - pub_approx
                print(f"  Q{q:<9} {rep_mean:>16.3f} {pub_approx:>14.3f} {diff:>10.3f}")

        ls_rep = quintile_df['LS'].mean() * 100
        ls_pub_approx = df['LS'].mean() * 300
        diff_ls = ls_rep - ls_pub_approx
        print(f"  {'─'*52}")
        print(f"  {'Q5-Q1':<10} {ls_rep:>16.3f} {ls_pub_approx:>14.3f} {diff_ls:>10.3f}")
        print(f"\n  Note: 'Pub×3' is a linear approximation. Geometric compounding")
        print(f"  and different sample periods will cause deviations.")

    return df


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replicate Weigert et al. (JF 2024) - "
                    "Unobserved Performance of Hedge Funds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replicate_up.py --demo
  python replicate_up.py --demo --validate data/UP_5-1.csv
  python replicate_up.py --demo --horizon 3
  python replicate_up.py --fund-returns data/fund_returns.csv \\
      --holdings data/holdings.csv --stock-returns data/stock_returns.csv \\
      --factors data/factors.csv --validate data/UP_5-1.csv
        """,
    )
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic demo data (no files needed).')
    parser.add_argument('--fund-returns', type=str,
                        help='Path to fund_returns.csv.')
    parser.add_argument('--holdings', type=str,
                        help='Path to holdings.csv.')
    parser.add_argument('--stock-returns', type=str,
                        help='Path to stock_returns.csv.')
    parser.add_argument('--factors', type=str,
                        help='Path to factors.csv.')
    parser.add_argument('--validate', type=str, default=None,
                        help='Path to published results CSV (e.g., UP_5-1.csv).')
    parser.add_argument('--horizon', type=int, default=1,
                        help='Prediction horizon in quarters (default=1).')
    parser.add_argument('--output-dir', type=str, default='data/results',
                        help='Output directory (default: data/results).')
    parser.add_argument('--n-funds', type=int, default=100,
                        help='Number of funds in demo mode (default: 100).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    return parser.parse_args()


# =============================================================================
# 10. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline."""
    args = parse_arguments()
    np.random.seed(args.seed)

    # ─── Step 1: Load Data ─────────────────────────────────────────────
    if args.demo:
        fund_returns_df, holdings_df, stock_returns_df, factors_df = \
            generate_demo_data(n_funds=args.n_funds)
    elif all([args.fund_returns, args.holdings, args.stock_returns, args.factors]):
        fund_returns_df, holdings_df, stock_returns_df, factors_df = \
            load_real_data(args.fund_returns, args.holdings,
                           args.stock_returns, args.factors)
    else:
        print("\n  ERROR: Provide all four data paths or use --demo.")
        print("  Run with --help for usage information.")
        sys.exit(1)

    # ─── Step 2: Compound to Quarterly ─────────────────────────────────
    fund_quarterly = compound_monthly_to_quarterly(fund_returns_df)
    stock_ret_wide = compound_stock_returns_quarterly(stock_returns_df)
    factors_quarterly = compound_factors_quarterly(factors_df)

    # ─── Step 3: Calculate UP Panel ────────────────────────────────────
    up_panel = calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide)

    if up_panel.empty:
        print("\n  FATAL: No valid UP observations. Check data alignment.")
        sys.exit(1)

    # ─── Step 4: Portfolio Sorts ───────────────────────────────────────
    ls_df, quintile_df = form_long_short_portfolio(
        up_panel, horizon=args.horizon, min_funds_per_quarter=MIN_FUNDS_PER_QUARTER
    )

    if ls_df.empty:
        print("\n  WARNING: Could not form long-short portfolio.")
    else:
        # ─── Step 5: Factor Model Alpha ────────────────────────────────
        compute_risk_adjusted_alpha(ls_df, factors_quarterly)

        # ─── Step 6: Quintile Analysis ─────────────────────────────────
        if not quintile_df.empty:
            analyze_quintile_returns(quintile_df, factors_quarterly)

    # ─── Step 7: Persistence ───────────────────────────────────────────
    analyze_persistence(up_panel)

    # ─── Step 8: Validate Against Published Results ────────────────────
    if args.validate:
        validate_path = Path(args.validate)
        if validate_path.exists():
            validate_against_paper(str(validate_path), quintile_df)
        else:
            print(f"\n  WARNING: Validation file not found: {args.validate}")

    # ─── Step 9: Save Results ──────────────────────────────────────────
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

    # ─── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  Replication Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
