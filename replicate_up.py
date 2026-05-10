"""
Unobserved Performance of Hedge Funds (Agarwal, Ruenzi & Weigert, JF 2024)
Python Replication Script - Full Version with Monthly/Quarterly Frequency Selection
Author: Tao Wu
Date: May 2026

## Key Features
- Selectable frequency: --freq monthly or --freq quarterly
- Full multi-fund panel processing with correct cross-sectional quintile sorting
- Accurate monthly-to-quarterly compounding (quarterly mode)
- Direct monthly computation with quarterly holdings alignment (monthly mode)
- Predictive long-short portfolios (t → t+horizon)
- Carhart 4-factor alphas with Newey-West HAC standard errors
- Built-in validation against the paper's official UP 5-1 benchmark
- Clean command-line interface (demo + real WRDS data)

## Frequency Modes

  --freq quarterly  (default)
      Quarterly UP calculation, sorting, and portfolio formation.
      Compounds monthly returns to quarterly. Holdings naturally align.
      Horizon is specified in quarters (default: 1 quarter ahead).

  --freq monthly
      Monthly UP calculation with quarterly holdings interpolated to monthly.
      Uses most recent 13F disclosure as of each month.
      Horizon is specified in months (default: 3 months ahead, matching paper).
      Newey-West lags automatically adjusted to 6 (monthly) vs 4 (quarterly).
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

__version__ = "3.0.0"

np.random.seed(42)

print("=" * 80)
print("Replicating Agarwal, Ruenzi & Weigert (JF 2024)")
print("Unobserved Performance of Hedge Funds")
print("Version 3.0: Monthly / Quarterly Frequency Selection")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_FUNDS_PER_PERIOD = 20
N_QUINTILES = 5
FACTOR_COLS = ['MKT', 'SMB', 'HML', 'MOM']

FREQ_CONFIG = {
    'monthly': {
        'label': 'Monthly',
        'periods_per_year': 12,
        'default_horizon': 3,       # t+3 months (as in paper)
        'default_nw_lags': 6,
        'period_col': 'month',
    },
    'quarterly': {
        'label': 'Quarterly',
        'periods_per_year': 4,
        'default_horizon': 1,       # t+1 quarter
        'default_nw_lags': 4,
        'period_col': 'quarter',
    },
}


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
# 2. DATA PREPARATION (Frequency-Dependent)
# =============================================================================

def compound_monthly_to_quarterly(fund_returns_df):
    """Compound monthly fund returns into quarterly returns."""
    print("\n  [Prep] Compounding fund returns to quarterly...")
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
    print(f"         → {len(quarterly):,} obs ({n_funds} funds × {n_quarters} quarters)")
    return quarterly


def compound_stock_returns_quarterly(stock_returns_df):
    """Compound monthly stock returns to quarterly (wide format)."""
    print("  [Prep] Compounding stock returns to quarterly...")
    df = stock_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby(['stock', 'quarter']).apply(
        lambda x: (1 + x['ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['stock', 'quarter', 'ret_q']

    stock_ret_wide = quarterly.pivot(index='quarter', columns='stock', values='ret_q')
    print(f"         → {stock_ret_wide.shape[0]} quarters × {stock_ret_wide.shape[1]} stocks")
    return stock_ret_wide


def compound_factors_quarterly(factors_df):
    """Compound monthly factors to quarterly."""
    print("  [Prep] Compounding factors to quarterly...")
    df = factors_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby('quarter')[FACTOR_COLS].apply(
        lambda x: (1 + x).prod() - 1
    )
    print(f"         → {len(quarterly)} quarters")
    return quarterly


def prepare_monthly_stock_returns(stock_returns_df):
    """Pivot monthly stock returns to wide format."""
    print("  [Prep] Pivoting stock returns to monthly wide format...")
    df = stock_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    stock_ret_wide = df.pivot_table(index='date', columns='stock', values='ret')
    print(f"         → {stock_ret_wide.shape[0]} months × {stock_ret_wide.shape[1]} stocks")
    return stock_ret_wide


def prepare_monthly_fund_returns(fund_returns_df):
    """Prepare monthly fund returns (minimal processing)."""
    print("  [Prep] Preparing monthly fund returns...")
    df = fund_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    n_funds = df['fund_id'].nunique()
    n_months = df['date'].nunique()
    print(f"         → {len(df):,} obs ({n_funds} funds × {n_months} months)")
    return df


def prepare_monthly_factors(factors_df):
    """Index monthly factors by date."""
    print("  [Prep] Preparing monthly factors...")
    df = factors_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')[FACTOR_COLS]
    print(f"         → {len(df)} months")
    return df


# =============================================================================
# 3. UP PANEL CALCULATION
# =============================================================================

def build_holdings_index(holdings_df):
    """Pre-group holdings by fund_id for O(1) lookup."""
    return {fid: group for fid, group in holdings_df.groupby('fund_id')}


def get_holdings_as_of(fund_id, as_of_date, holdings_by_fund):
    """
    Get the most recent quarterly holdings disclosure on or before as_of_date.

    Returns
    -------
    DataFrame with columns ['stock', 'weight'] or None
    """
    if fund_id not in holdings_by_fund:
        return None

    fund_holdings = holdings_by_fund[fund_id]
    prev_reports = fund_holdings[fund_holdings['report_date'] <= as_of_date]

    if prev_reports.empty:
        return None

    latest_date = prev_reports['report_date'].max()
    latest_holdings = prev_reports[prev_reports['report_date'] == latest_date].copy()

    total_weight = latest_holdings['weight'].sum()
    if total_weight <= 0:
        return None
    latest_holdings['weight'] = latest_holdings['weight'] / total_weight

    return latest_holdings[['stock', 'weight']]


def compute_holdings_return(holdings_slice, stock_ret_row):
    """
    Compute buy-and-hold return for given holdings and stock returns.

    Parameters
    ----------
    holdings_slice : DataFrame with ['stock', 'weight']
    stock_ret_row : Series indexed by stock with return values

    Returns
    -------
    float or np.nan
    """
    common_stocks = [s for s in holdings_slice['stock'].values
                     if s in stock_ret_row.index and not pd.isna(stock_ret_row[s])]

    if len(common_stocks) == 0:
        return np.nan

    weights = holdings_slice.set_index('stock').loc[common_stocks, 'weight']
    weights = weights / weights.sum()
    returns = stock_ret_row[common_stocks]

    return float((weights * returns).sum())


# ─── Quarterly UP ───────────────────────────────────────────────────────────

def calculate_up_panel_quarterly(fund_quarterly, holdings_df, stock_ret_wide):
    """
    Calculate UP panel at quarterly frequency.

    UP(i,t) = R_reported(i,t) - R_buyandhold(i,t)
    Holdings from quarter t-1 applied to stock returns in quarter t.
    """
    print("\n  [Step 3] Computing Unobserved Performance (UP) — QUARTERLY")
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

            # Get holdings from the quarter *before* the evaluation quarter
            holdings_slice = get_holdings_as_of(fund_id, q - pd.Timedelta(days=1),
                                                holdings_by_fund)
            if holdings_slice is None:
                continue

            if q not in stock_ret_wide.index:
                continue

            bh_ret = compute_holdings_return(holdings_slice, stock_ret_wide.loc[q])
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
    _print_up_summary(up_panel, 'quarter')
    return up_panel


# ─── Monthly UP ────────────────────────────────────────────────────────────

def calculate_up_panel_monthly(fund_monthly_df, holdings_df, stock_ret_wide_monthly):
    """
    Calculate UP panel at monthly frequency.

    UP(i,t) = R_reported(i,t) - R_buyandhold(i,t)
    Holdings from the most recent quarter-end on or before month t-1,
    applied to stock returns in month t.

    This follows the paper's approach: use last-available 13F to construct
    a hypothetical portfolio return each month.
    """
    print("\n  [Step 3] Computing Unobserved Performance (UP) — MONTHLY")
    print("           Building holdings index...")
    holdings_by_fund = build_holdings_index(holdings_df)

    fund_ids = fund_monthly_df['fund_id'].unique()
    total = len(fund_ids)
    progress_step = max(1, total // 5)

    results = []

    for idx, fund_id in enumerate(fund_ids):
        if (idx + 1) % progress_step == 0:
            pct = (idx + 1) / total * 100
            print(f"           Progress: {idx + 1:>5}/{total} funds ({pct:.0f}%)")

        fund_data = fund_monthly_df[fund_monthly_df['fund_id'] == fund_id]

        for _, row in fund_data.iterrows():
            month = row['date']
            reported_ret = row['fund_ret']

            # Use holdings from the most recent quarter-end *before* current month
            # (i.e., holdings known at the start of month t)
            holdings_slice = get_holdings_as_of(
                fund_id, month - pd.Timedelta(days=1), holdings_by_fund
            )
            if holdings_slice is None:
                continue

            if month not in stock_ret_wide_monthly.index:
                continue

            bh_ret = compute_holdings_return(
                holdings_slice, stock_ret_wide_monthly.loc[month]
            )
            if pd.isna(bh_ret):
                continue

            results.append({
                'month': month,
                'fund_id': fund_id,
                'UP': reported_ret - bh_ret,
                'fund_ret_m': reported_ret,
                'bh_ret_m': bh_ret,
            })

    up_panel = pd.DataFrame(results)
    _print_up_summary(up_panel, 'month')
    return up_panel


def _print_up_summary(up_panel, period_col):
    """Print summary statistics for UP panel."""
    if up_panel.empty:
        print("           ⚠  WARNING: No valid UP observations computed.")
        return

    n_obs = len(up_panel)
    n_funds = up_panel['fund_id'].nunique()
    n_periods = up_panel[period_col].nunique()
    freq_label = "quarters" if period_col == "quarter" else "months"
    print(f"           → {n_obs:,} observations ({n_funds} funds, "
          f"{n_periods} {freq_label})")
    print(f"           → UP mean = {up_panel['UP'].mean():.5f}, "
          f"std = {up_panel['UP'].std():.5f}, "
          f"median = {up_panel['UP'].median():.5f}")


# =============================================================================
# 4. PREDICTIVE LONG-SHORT PORTFOLIO
# =============================================================================

def form_long_short_portfolio(up_panel, freq='quarterly', horizon=None,
                              min_funds=MIN_FUNDS_PER_PERIOD):
    """
    Cross-sectional quintile sort on UP(t) → t+horizon actual fund returns.

    Parameters
    ----------
    up_panel : DataFrame
        Must contain columns: [period_col, 'fund_id', 'UP', return_col]
    freq : str
        'monthly' or 'quarterly'
    horizon : int
        Number of periods ahead for evaluation. If None, uses default from config.
    min_funds : int
        Minimum funds required per sort period.

    Returns
    -------
    tuple (ls_df, quintile_df)
    """
    config = FREQ_CONFIG[freq]
    if horizon is None:
        horizon = config['default_horizon']

    period_col = config['period_col']
    return_col = 'fund_ret_m' if freq == 'monthly' else 'fund_ret_q'

    print(f"\n  [Step 4] Forming Long-Short Portfolio "
          f"({config['label']}, horizon = t+{horizon})")

    periods = sorted(up_panel[period_col].unique())
    ls_results = []
    quintile_results = []
    skipped = 0

    for i in range(len(periods) - horizon):
        sort_period = periods[i]
        eval_period = periods[i + horizon]

        # Cross-sectional sort on UP in period t
        sort_data = up_panel[up_panel[period_col] == sort_period].copy()

        if len(sort_data) < min_funds:
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
        eval_data = up_panel[up_panel[period_col] == eval_period][
            ['fund_id', return_col]
        ].rename(columns={return_col: 'eval_return'})

        merged = sort_data[['fund_id', 'quintile']].merge(
            eval_data, on='fund_id', how='inner'
        )

        if len(merged) < min_funds:
            skipped += 1
            continue

        # Equal-weighted portfolio returns per quintile
        port_rets = merged.groupby('quintile')['eval_return'].mean()

        if N_QUINTILES in port_rets.index and 1 in port_rets.index:
            ls_ret = port_rets[N_QUINTILES] - port_rets[1]
            ls_results.append({period_col: eval_period, 'LS_return': ls_ret})

            row = {period_col: eval_period}
            for q_bin in range(1, N_QUINTILES + 1):
                row[f'Q{q_bin}'] = port_rets.get(q_bin, np.nan)
            row['LS'] = ls_ret
            row['n_funds'] = len(merged)
            quintile_results.append(row)

    ls_df = pd.DataFrame(ls_results)
    if not ls_df.empty:
        ls_df = ls_df.set_index(period_col)

    quintile_df = pd.DataFrame(quintile_results)
    if not quintile_df.empty:
        quintile_df = quintile_df.set_index(period_col)

    n_valid = len(ls_df)
    print(f"           → {n_valid} evaluation periods (skipped {skipped})")

    if n_valid > 0:
        mean_ls = ls_df['LS_return'].mean()
        std_ls = ls_df['LS_return'].std(ddof=1)
        t_simple = mean_ls / (std_ls / np.sqrt(n_valid)) if std_ls > 0 else np.nan
        unit = "month" if freq == 'monthly' else "quarter"
        print(f"           → Mean L/S return: {mean_ls*100:.3f}% per {unit}")
        print(f"           → Std:             {std_ls*100:.3f}%")
        print(f"           → t-statistic:     {t_simple:.2f}")

    return ls_df, quintile_df


# =============================================================================
# 5. RISK-ADJUSTED ALPHA (Carhart 4-Factor, Newey-West)
# =============================================================================

def compute_risk_adjusted_alpha(ls_df, factors_indexed, freq='quarterly',
                                max_lags=None):
    """
    Carhart 4-factor regression with Newey-West standard errors.

    Model: LS_t = alpha + b1*MKT_t + b2*SMB_t + b3*HML_t + b4*MOM_t + eps_t

    Parameters
    ----------
    ls_df : DataFrame
        Indexed by period (month or quarter), with 'LS_return' column.
    factors_indexed : DataFrame
        Indexed by period (same freq), with FACTOR_COLS columns.
    freq : str
        'monthly' or 'quarterly'
    max_lags : int or None
        Newey-West lags. If None, uses default from FREQ_CONFIG.
    """
    config = FREQ_CONFIG[freq]
    if max_lags is None:
        max_lags = config['default_nw_lags']

    print(f"\n  [Step 5] Carhart 4-Factor Alpha "
          f"({config['label']}, Newey-West {max_lags} lags)")
    print(f"  {'─'*56}")

    combined = ls_df[['LS_return']].join(factors_indexed, how='inner').dropna()

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

    alpha = model.params['const']
    t_stat = model.tvalues['const']
    ppy = config['periods_per_year']
    alpha_ann = alpha * ppy
    unit = "month" if freq == 'monthly' else "quarter"

    print(f"\n  Key Result:")
    print(f"    {config['label']} alpha = {alpha*100:.3f}% per {unit}")
    print(f"    Annualized alpha   = {alpha_ann*100:.2f}%")
    print(f"    t-statistic        = {t_stat:.3f}", end="")
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
        'alpha': alpha,
        'alpha_annualized': alpha_ann,
        't_stat': t_stat,
        'p_value': model.pvalues['const'],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'freq': freq,
    }


# =============================================================================
# 6. QUINTILE MONOTONICITY ANALYSIS
# =============================================================================

def analyze_quintile_returns(quintile_df, factors_indexed, freq='quarterly',
                             max_lags=None):
    """
    Analyze whether portfolio returns increase monotonically from Q1 to Q5.
    Estimates individual alpha for each quintile portfolio.
    """
    config = FREQ_CONFIG[freq]
    if max_lags is None:
        max_lags = config['default_nw_lags']

    unit = "mo" if freq == 'monthly' else "qtr"
    ppy = config['periods_per_year']

    print(f"\n  [Step 6] Quintile Portfolio Analysis ({config['label']})")
    print(f"  {'═'*70}")
    print(f"  {'Portfolio':<12} {'Mean%/'+unit:>10} {'Ann%':>8} "
          f"{'Std%/'+unit:>10} {'Alpha%':>8} {'t(α)':>7}")
    print(f"  {'─'*70}")

    summary_rows = []

    for q_bin in range(1, N_QUINTILES + 1):
        col = f"Q{q_bin}"
        if col not in quintile_df.columns:
            continue

        rets = quintile_df[col].dropna()
        mean_ret = rets.mean()
        std_ret = rets.std(ddof=1)
        ann_ret = mean_ret * ppy

        # Compute alpha for each quintile
        alpha, t_alpha = np.nan, np.nan
        combined = rets.to_frame('ret').join(factors_indexed, how='inner').dropna()
        if len(combined) >= 12:
            y_q = combined['ret']
            X_q = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y_q, X_q).fit(
                cov_type='HAC', cov_kwds={'maxlags': max_lags}
            )
            alpha = mod.params['const']
            t_alpha = mod.tvalues['const']

        label = (f"Q{q_bin} (Low)" if q_bin == 1
                 else f"Q{q_bin} (High)" if q_bin == N_QUINTILES
                 else f"Q{q_bin}")
        print(
            f"  {label:<12} {mean_ret*100:>10.3f} {ann_ret*100:>8.2f} "
            f"{std_ret*100:>10.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )
        summary_rows.append({
            'quintile': q_bin,
            'mean': mean_ret,
            'annualized': ann_ret,
            'std': std_ret,
            'alpha': alpha,
            't_alpha': t_alpha,
        })

    # Long-Short row
    if 'LS' in quintile_df.columns:
        rets = quintile_df['LS'].dropna()
        mean_ret = rets.mean()
        std_ret = rets.std(ddof=1)
        combined = rets.to_frame('ret').join(factors_indexed, how='inner').dropna()
        alpha, t_alpha = np.nan, np.nan
        if len(combined) >= 12:
            y_q = combined['ret']
            X_q = sm.add_constant(combined[FACTOR_COLS])
            mod = sm.OLS(y_q, X_q).fit(
                cov_type='HAC', cov_kwds={'maxlags': max_lags}
            )
            alpha = mod.params['const']
            t_alpha = mod.tvalues['const']
        print(f"  {'─'*70}")
        print(
            f"  {'Q5-Q1':<12} {mean_ret*100:>10.3f} {mean_ret*ppy*100:>8.2f} "
            f"{std_ret*100:>10.3f} {alpha*100:>8.3f} {t_alpha:>7.2f}"
        )

    print(f"  {'═'*70}")
    return pd.DataFrame(summary_rows)


# =============================================================================
# 7. UP PERSISTENCE ANALYSIS (Fama-MacBeth)
# =============================================================================

def analyze_persistence(up_panel, freq='quarterly'):
    """
    Test UP persistence using Fama-MacBeth cross-sectional regressions.
    If UP reflects persistent skill, UP(t-1) should predict UP(t).
    """
    config = FREQ_CONFIG[freq]
    period_col = config['period_col']

    print(f"\n  [Step 7] UP Persistence Analysis "
          f"(Fama-MacBeth, {config['label']})")
    print(f"  {'─'*50}")

    panel = up_panel[[period_col, 'fund_id', 'UP']].copy()
    panel = panel.sort_values(['fund_id', period_col])
    panel['UP_lag1'] = panel.groupby('fund_id')['UP'].shift(1)

    valid = panel.dropna(subset=['UP', 'UP_lag1'])

    if len(valid) < 50:
        print("           ⚠  Insufficient data for persistence test.")
        return {}

    # Fama-MacBeth: cross-sectional regression each period
    periods = sorted(valid[period_col].unique())
    coeffs = []

    for p in periods:
        p_data = valid[valid[period_col] == p]
        if len(p_data) < 10:
            continue
        X = sm.add_constant(p_data['UP_lag1'])
        y = p_data['UP']
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

def validate_against_paper(csv_path, quintile_df=None, freq='quarterly'):
    """
    Load published portfolio sort results and compare with replication.

    The validation CSV contains monthly returns for quintile portfolios
    sorted on UP(t) with evaluation in t+3, as reported in the paper.
    """
    config = FREQ_CONFIG[freq]

    print(f"\n  {'═'*65}")
    print(f"  VALIDATION: Published Results from Agarwal et al. (JF 2024)")
    print(f"  Replication frequency: {config['label']}")
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

    print(f"\n  Published Sample: {date_range} ({n_months} months)")
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
        if freq == 'quarterly':
            print(f"  Comparison: Replicated (quarterly) vs. Published (monthly × 3)")
            scale = 3.0
            rep_unit = "%/qtr"
        else:
            print(f"  Comparison: Replicated (monthly) vs. Published (monthly)")
            scale = 1.0
            rep_unit = "%/mo"

        print(f"  {'Portfolio':<10} {'Replicated':>16} {'Published':>14} {'Diff':>10}")
        print(f"  {'':<10} {rep_unit:>16} {'%/mo×'+str(int(scale)):>14} {'':<10}")
        print(f"  {'─'*52}")

        for q in range(1, 6):
            col_rep = f"Q{q}"
            col_pub = f"PF{q}"
            if col_rep in quintile_df.columns:
                rep_mean = quintile_df[col_rep].mean() * 100
                pub_approx = df[col_pub].mean() * 100 * scale
                diff = rep_mean - pub_approx
                print(f"  Q{q:<9} {rep_mean:>16.3f} {pub_approx:>14.3f} {diff:>10.3f}")

        ls_rep = quintile_df['LS'].mean() * 100
        ls_pub_approx = df['LS'].mean() * 100 * scale
        diff_ls = ls_rep - ls_pub_approx
        print(f"  {'─'*52}")
        print(f"  {'Q5-Q1':<10} {ls_rep:>16.3f} {ls_pub_approx:>14.3f} {diff_ls:>10.3f}")

        if freq == 'quarterly':
            print(f"\n  Note: 'Published ×3' is a linear approximation. Geometric")
            print(f"  compounding and sample-period differences cause deviations.")
        else:
            print(f"\n  Note: Differences may arise from sample period, data coverage,")
            print(f"  and exact sorting methodology.")

    return df


# =============================================================================
# 9. COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replicate Agarwal, Ruenzi & Weigert (JF 2024) — "
                    "Unobserved Performance of Hedge Funds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quarterly mode (default)
  python replicate_up.py --demo --freq quarterly

  # Monthly mode with t+3 horizon (matching paper)
  python replicate_up.py --demo --freq monthly --horizon 3

  # Quarterly with validation
  python replicate_up.py --demo --freq quarterly --validate data/UP_5-1.csv

  # Real WRDS data, monthly
  python replicate_up.py --freq monthly --horizon 3 \\
      --fund-returns data/fund_returns.csv \\
      --holdings data/holdings.csv \\
      --stock-returns data/stock_returns.csv \\
      --factors data/factors.csv \\
      --validate data/UP_5-1.csv
        """,
    )
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic demo data (no files needed).')
    parser.add_argument('--freq', type=str, choices=['monthly', 'quarterly'],
                        default='quarterly',
                        help='Analysis frequency: monthly or quarterly (default: quarterly).')
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
    parser.add_argument('--horizon', type=int, default=None,
                        help='Prediction horizon in periods. '
                             'Default: 1 (quarterly) or 3 (monthly).')
    parser.add_argument('--nw-lags', type=int, default=None,
                        help='Newey-West lags. Default: 4 (quarterly) or 6 (monthly).')
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

    freq = args.freq
    config = FREQ_CONFIG[freq]
    horizon = args.horizon if args.horizon is not None else config['default_horizon']
    nw_lags = args.nw_lags if args.nw_lags is not None else config['default_nw_lags']

    print(f"\n  Configuration:")
    print(f"    Frequency:       {config['label']}")
    print(f"    Horizon:         t + {horizon} {config['label'].lower()} periods")
    print(f"    Newey-West lags: {nw_lags}")
    print(f"    Mode:            {'Demo' if args.demo else 'Real data'}")

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

    # ─── Step 2: Prepare Data (frequency-dependent) ────────────────────
    if freq == 'quarterly':
        fund_prepared = compound_monthly_to_quarterly(fund_returns_df)
        stock_ret_wide = compound_stock_returns_quarterly(stock_returns_df)
        factors_indexed = compound_factors_quarterly(factors_df)
    else:  # monthly
        fund_prepared = prepare_monthly_fund_returns(fund_returns_df)
        stock_ret_wide = prepare_monthly_stock_returns(stock_returns_df)
        factors_indexed = prepare_monthly_factors(factors_df)

    # ─── Step 3: Calculate UP Panel ────────────────────────────────────
    if freq == 'quarterly':
        up_panel = calculate_up_panel_quarterly(
            fund_prepared, holdings_df, stock_ret_wide
        )
    else:  # monthly
        up_panel = calculate_up_panel_monthly(
            fund_prepared, holdings_df, stock_ret_wide
        )

    if up_panel.empty:
        print("\n  FATAL: No valid UP observations. Check data alignment.")
        sys.exit(1)

    # ─── Step 4: Portfolio Sorts ───────────────────────────────────────
    ls_df, quintile_df = form_long_short_portfolio(
        up_panel, freq=freq, horizon=horizon, min_funds=MIN_FUNDS_PER_PERIOD
    )

    if ls_df.empty:
        print("\n  WARNING: Could not form long-short portfolio.")
    else:
        # ─── Step 5: Factor Model Alpha ────────────────────────────────
        compute_risk_adjusted_alpha(ls_df, factors_indexed, freq=freq,
                                    max_lags=nw_lags)

        # ─── Step 6: Quintile Analysis ─────────────────────────────────
        if not quintile_df.empty:
            analyze_quintile_returns(quintile_df, factors_indexed, freq=freq,
                                     max_lags=nw_lags)

    # ─── Step 7: Persistence ───────────────────────────────────────────
    analyze_persistence(up_panel, freq=freq)

    # ─── Step 8: Validate Against Published Results ────────────────────
    if args.validate:
        validate_path = Path(args.validate)
        if validate_path.exists():
            validate_against_paper(str(validate_path), quintile_df, freq=freq)
        else:
            print(f"\n  WARNING: Validation file not found: {args.validate}")

    # ─── Step 9: Save Results ──────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{freq}"
    up_panel.to_csv(output_dir / f"up_panel{suffix}.csv", index=False)
    print(f"\n  Saved: {output_dir / f'up_panel{suffix}.csv'}")

    if not ls_df.empty:
        ls_df.to_csv(output_dir / f"long_short_returns{suffix}.csv")
        print(f"  Saved: {output_dir / f'long_short_returns{suffix}.csv'}")

    if not quintile_df.empty:
        quintile_df.to_csv(output_dir / f"quintile_returns{suffix}.csv")
        print(f"  Saved: {output_dir / f'quintile_returns{suffix}.csv'}")

    # ─── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  Replication Complete ({config['label']} Frequency)")
    print("=" * 80)


if __name__ == "__main__":
    main()
