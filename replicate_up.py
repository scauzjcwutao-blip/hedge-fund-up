"""
Unobserved Performance of Hedge Funds (Weigert et al., Journal of Finance 2024)
Python Replication Script - Final Corrected Full Version
Author: Tao Wu
Date: May 2026

Key Features:
  • Multi-fund panel processing (correct cross-sectional sorting)
  • Monthly returns correctly compounded to quarterly
  • Cross-sectional quintile sorting on UP(t) each quarter
  • Predictive sort: t-period UP → t+1 actual fund returns
  • Newey-West (HAC) standard errors
  • Optimized holdings indexing for high performance
  • Demo mode + full WRDS data support
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

np.random.seed(42)

print("=" * 80)
print("Replicating Weigert et al. (JF 2024) - Unobserved Performance of Hedge Funds")
print("Final Corrected Version: Multi-Fund Panel + Predictive Sort")
print("=" * 80)


# =============================================================================
# CSV FILE FORMAT REQUIREMENTS
# =============================================================================
"""
1. fund_returns.csv
   - Columns: 'date', 'fund_id', 'fund_ret' (monthly decimal)

2. holdings.csv
   - Columns: 'report_date', 'fund_id', 'stock', 'weight'

3. stock_returns.csv
   - Columns: 'date', 'stock', 'ret' (monthly decimal)

4. factors.csv
   - Columns: 'date', 'MKT', 'SMB', 'HML', 'MOM' (monthly decimal)
"""


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def generate_demo_data(n_funds=100, n_stocks=50):
    """Generate realistic demo data for testing."""
    print(f"\n📊 Generating demo data: {n_funds} funds, {n_stocks} stocks")
    months = pd.date_range(start='1994-01-31', end='2019-12-31', freq='ME')
    quarters = pd.date_range(start='1994-03-31', end='2019-12-31', freq='QE')
    fund_ids = [f"FUND{i:04d}" for i in range(1, n_funds + 1)]
    stock_ids = [f"STK{i:03d}" for i in range(1, n_stocks + 1)]

    # Fund monthly returns
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
                    'report_date': q, 'fund_id': fid, 'stock': stock, 'weight': weight
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

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


def load_real_data():
    """Load real WRDS CSV data."""
    print("\n📂 Loading real data from CSV files...")
    fund_path = input("   Path to fund_returns.csv: ").strip()
    fund_returns_df = pd.read_csv(fund_path, parse_dates=['date'])

    holdings_path = input("   Path to holdings.csv: ").strip()
    holdings_df = pd.read_csv(holdings_path, parse_dates=['report_date'])

    stock_path = input("   Path to stock_returns.csv: ").strip()
    stock_returns_df = pd.read_csv(stock_path, parse_dates=['date'])

    factor_path = input("   Path to factors.csv: ").strip()
    factors_df = pd.read_csv(factor_path, parse_dates=['date'])

    return fund_returns_df, holdings_df, stock_returns_df, factors_df


def load_data():
    """Main data loading dispatcher."""
    print("\n🔄 Data Loading Mode")
    choice = input("   Use [D]emo data or [R]eal CSV data? (D/R, default=D): ").strip().upper()
    if choice == 'R':
        return load_real_data()
    else:
        return generate_demo_data()


# =============================================================================
# 2. COMPOUND MONTHLY TO QUARTERLY
# =============================================================================

def compound_monthly_to_quarterly(fund_returns_df):
    """Compound monthly fund returns into quarterly returns."""
    print("\n🔄 Compounding monthly returns to quarterly...")
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

    print(f"   ✅ Quarterly fund returns: {len(quarterly)} obs")
    return quarterly


def compound_stock_returns_quarterly(stock_returns_df):
    """Compound monthly stock returns to quarterly (wide format)."""
    print("\n🔄 Compounding stock returns to quarterly...")
    df = stock_returns_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    quarterly = df.groupby(['stock', 'quarter']).apply(
        lambda x: (1 + x['ret']).prod() - 1, include_groups=False
    ).reset_index()
    quarterly.columns = ['stock', 'quarter', 'ret_q']

    stock_ret_wide = quarterly.pivot(index='quarter', columns='stock', values='ret_q')
    print(f"   ✅ Quarterly stock returns: {stock_ret_wide.shape}")
    return stock_ret_wide


def compound_factors_quarterly(factors_df):
    """Compound monthly factors to quarterly."""
    print("\n🔄 Compounding factors to quarterly...")
    df = factors_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['quarter'] = df['date'].dt.to_period('Q').dt.to_timestamp('Q')

    factor_cols = ['MKT', 'SMB', 'HML', 'MOM']
    quarterly = df.groupby('quarter')[factor_cols].apply(
        lambda x: (1 + x).prod() - 1
    )
    print(f"   ✅ Quarterly factors: {len(quarterly)} quarters")
    return quarterly


# =============================================================================
# 3. OPTIMIZED UP PANEL CALCULATION
# =============================================================================

def build_holdings_index(holdings_df: pd.DataFrame) -> dict:
    """Pre-group holdings by fund_id for major performance improvement."""
    print("   🔧 Building holdings index by fund_id...")
    return {fid: group for fid, group in holdings_df.groupby('fund_id')}


def construct_buy_and_hold_return(fund_id, quarter, holdings_by_fund: dict, stock_ret_wide):
    """Optimized buy-and-hold return using pre-grouped index."""
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

    return (weights * returns).sum()


def calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide):
    """Calculate UP panel for all funds with optimized holdings lookup."""
    print("\n🔄 Calculating Unobserved Performance (UP) for all funds...")
    holdings_by_fund = build_holdings_index(holdings_df)

    quarters = sorted(fund_quarterly['quarter'].unique())
    fund_ids = fund_quarterly['fund_id'].unique()

    results = []
    total = len(fund_ids)
    progress_step = max(1, total // 10)

    for idx, fund_id in enumerate(fund_ids):
        if (idx + 1) % progress_step == 0:
            print(f"   Processing fund {idx + 1}/{total}...")

        fund_data = fund_quarterly[fund_quarterly['fund_id'] == fund_id]
        for _, row in fund_data.iterrows():
            q = row['quarter']
            reported_ret = row['fund_ret_q']
            bh_ret = construct_buy_and_hold_return(fund_id, q, holdings_by_fund, stock_ret_wide)

            if pd.isna(bh_ret):
                continue

            up = reported_ret - bh_ret
            results.append({
                'quarter': q,
                'fund_id': fund_id,
                'UP': up,
                'fund_ret_q': reported_ret
            })

    up_panel = pd.DataFrame(results)
    print(f"   ✅ UP Panel: {len(up_panel)} observations ({up_panel['fund_id'].nunique()} funds)")
    return up_panel


# =============================================================================
# 4. PREDICTIVE LONG-SHORT PORTFOLIO
# =============================================================================

def form_long_short_portfolio(up_panel, min_funds_per_quarter=20):
    """Cross-sectional quintile sort on UP(t) → t+1 actual fund returns."""
    print("\n🔄 Forming Long-Short Portfolio (Predictive Sort)...")
    quarters = sorted(up_panel['quarter'].unique())
    ls_results = []
    skipped = 0

    for i in range(len(quarters) - 1):
        current_q = quarters[i]
        next_q = quarters[i + 1]

        current_data = up_panel[up_panel['quarter'] == current_q].copy()
        if len(current_data) < min_funds_per_quarter:
            skipped += 1
            continue

        try:
            current_data['quintile'] = pd.qcut(
                current_data['UP'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
            ).astype(int)
        except ValueError:
            skipped += 1
            continue

        next_data = up_panel[up_panel['quarter'] == next_q][['fund_id', 'fund_ret_q']]
        merged = current_data[['fund_id', 'quintile']].merge(next_data, on='fund_id', how='inner')

        if len(merged) < min_funds_per_quarter:
            skipped += 1
            continue

        port_returns = merged.groupby('quintile')['fund_ret_q'].mean()

        if 5 in port_returns.index and 1 in port_returns.index:
            ls_ret = port_returns[5] - port_returns[1]
            ls_results.append({'quarter': next_q, 'LS_return': ls_ret})

    ls_df = pd.DataFrame(ls_results).set_index('quarter')
    print(f"   ✅ Long-Short portfolio: {len(ls_df)} quarters (skipped {skipped})")
    return ls_df


# =============================================================================
# 5. RISK-ADJUSTED ALPHA (Carhart 4-Factor, Newey-West)
# =============================================================================

def compute_risk_adjusted_alpha(ls_df, factors_quarterly, max_lags=4):
    """Carhart 4-factor regression with Newey-West standard errors."""
    print("\n🔄 Computing Risk-Adjusted Alpha (Carhart 4-Factor, Newey-West)...")
    combined = ls_df[['LS_return']].join(factors_quarterly, how='inner').dropna()

    if len(combined) < 12:
        print("   ⚠️  Insufficient observations for regression.")
        return None

    y = combined['LS_return']
    X = sm.add_constant(combined[['MKT', 'SMB', 'HML', 'MOM']])
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})

    alpha = model.params['const']
    t_stat = model.tvalues['const']
    print(f"   Alpha (quarterly) = {alpha:.4f} (t = {t_stat:.2f})")
    return {'alpha_quarterly': alpha, 't_stat': t_stat, 'n_obs': len(combined)}
# =============================================================================
# 6. VALIDATION AGAINST PAPER 
# =============================================================================
def validate_against_paper(
    csv_path: str,
    quintile_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Load and summarize the published portfolio sort results for comparison.
    """
    print(f"\n  {'═'*65}")
    print(f"  VALIDATION: Published Results from Weigert et al. (JF 2024)")
    print(f"  {'═'*65}")

    df = pd.read_csv(csv_path, skiprows=1)

    # Standardize column names
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

    # ====================== 新增：与自己复现结果直接对比 ======================
    if quintile_df is not None and not quintile_df.empty and 'LS' in quintile_df.columns:
        print("\n  📊 Direct Comparison with Replicated Results:")
        print(f"  {'Portfolio':<12} {'Replicated (Quarterly)':>22} {'Published (t+3 monthly)':>24} {'Diff':>10}")
        print(f"  {'─'*70}")

        for q in range(1, 6):
            col_rep = f"Q{q}"
            col_pub = f"PF{q}"
            
            if col_rep in quintile_df.columns and col_pub in df.columns:
                rep_mean = quintile_df[col_rep].mean() * 100          # quarterly %
                pub_mean_monthly = df[col_pub].mean() * 100           # monthly %
                pub_mean_quarterly_approx = pub_mean_monthly * 3       # 粗略季度化
                
                diff = rep_mean - pub_mean_quarterly_approx
                print(f"  Q{q:<11} {rep_mean:>22.3f}% {pub_mean_quarterly_approx:>24.3f}% {diff:>10.3f}%")

        # Long-Short 对比
        if 'LS' in quintile_df.columns and 'LS' in df.columns:
            ls_rep = quintile_df['LS'].mean() * 100
            ls_pub_monthly = df['LS'].mean() * 100
            ls_pub_quarterly_approx = ls_pub_monthly * 3
            diff_ls = ls_rep - ls_pub_quarterly_approx
            print(f"  {'─'*70}")
            print(f"  Q5-Q1      {ls_rep:>22.3f}% {ls_pub_quarterly_approx:>24.3f}% {diff_ls:>10.3f}%")

    return df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    fund_returns_df, holdings_df, stock_returns_df, factors_df = load_data()

    fund_quarterly = compound_monthly_to_quarterly(fund_returns_df)
    stock_ret_wide = compound_stock_returns_quarterly(stock_returns_df)
    factors_quarterly = compound_factors_quarterly(factors_df)

    up_panel = calculate_up_panel(fund_quarterly, holdings_df, stock_ret_wide)

    if up_panel.empty:
        print("\n❌ No valid UP observations.")
        exit()

    ls_df = form_long_short_portfolio(up_panel, min_funds_per_quarter=20)

    if not ls_df.empty:
        compute_risk_adjusted_alpha(ls_df, factors_quarterly)

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    up_panel.to_csv(output_dir / "up_panel.csv", index=False)
    ls_df.to_csv(output_dir / "long_short_returns.csv")

    print("\n" + "=" * 80)
    print("🎉 Replication Completed Successfully!")
    print(f"📁 Results saved to: {output_dir.resolve()}")
    print("=" * 80)
