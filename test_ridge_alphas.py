"""
Grid search script to test different Ridge alpha values.
Tests multiple regularization strengths and compares performance metrics.

Usage:
    python test_ridge_alphas.py

This script tests different Ridge regularization alphas by updating config
and running individual backtests. Results are saved to ridge_grid_search_results.csv
"""

import pandas as pd
import numpy as np
import sys
import gc
import subprocess
import os

# Ridge alphas to test
RIDGE_ALPHAS_TO_TEST = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]


def run_single_alpha_backtest(ridge_alpha):
    """
    Run backtest for a single Ridge alpha value.
    Extracts only metrics - NO PLOTTING or verbose output.
    """
    print(f"  Testing α={ridge_alpha}...", end=" ", flush=True)
    
    try:
        # Create a temporary Python script that runs backtest metrics only
        temp_script = f"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# Update config BEFORE importing other modules
import config
config.RIDGE_ALPHA = {ridge_alpha}

from data_loader import (load_sector_etf_prices, load_signals, load_ff5_mom_monthly, 
                        load_vix_monthly, load_credit_spread_monthly)
from signal_processing import (etf_monthly_returns, apply_inception_mask, monthly_signal_panel, 
                              zscore_signals, monthly_volatility)
from exposure_model import rolling_sectors_betas, sector_expected_returns
from portfolio_construction import (rank_and_weight_from_forecast_hysteresis, 
                                   apply_position_caps_and_renormalize, scale_to_target_vol)
from crisis_filter import compute_crisis_flag, apply_crisis_reduction
from backtest import simulate_portfolio
from benchmark import align_factors
import numpy as np

# Load data
etf_px = load_sector_etf_prices()
sigs_raw = load_signals()
vix_m = load_vix_monthly()
cs_m = load_credit_spread_monthly()

r_full = etf_monthly_returns(etf_px)
r_full = apply_inception_mask(r_full, config.ETF_INCEPTION)

crisis_flag_full = compute_crisis_flag(vix=vix_m, credit_spread=cs_m,
    window_months=config.CRISIS_WINDOW_MONTHS, p_vix=config.CRISIS_VIX_PERCENTILE, 
    p_cs=config.CRISIS_CS_PERCENTILE, logic=config.CRISIS_LOGIC, 
    min_on=config.CRISIS_MIN_ON, min_off=config.CRISIS_MIN_OFF)

sig_full = monthly_signal_panel(sigs_raw)
z_full = zscore_signals(sig_full)

betas = rolling_sectors_betas(r_full, z_full, lookback_months=config.LOOKBACK_BETA_MONTHS, 
                             min_months=config.MIN_BETA_MONTHS)
fcast_full = sector_expected_returns(betas, z_full)
vol_full = monthly_volatility(r_full, config.LOOKBACK_VOL_MONTHS)

weights_full, _ = rank_and_weight_from_forecast_hysteresis(
    fcast_full, r_full, entry_long=3, exit_long=6, dwell_min=2, min_bucket_size=4)

weights_full = apply_position_caps_and_renormalize(weights_full, L_target=0.2, S_target=0.2,
                                                   max_long_single=0.25, max_short_single=0.15)
weights_full = apply_crisis_reduction(weights=weights_full, crisis_flag=crisis_flag_full,
                                     reduction=config.CRISIS_REDUCTION)
weights_full = scale_to_target_vol(sector_returns=r_full, weights=weights_full, vol_target=0.12,
                                  lookback_months=12, smoothing_window=3, crisis_flag=crisis_flag_full)

r_m = r_full.loc[config.BACKTEST_START:config.BACKTEST_END]
weights = weights_full.loc[config.BACKTEST_START:config.BACKTEST_END]

port_rets = simulate_portfolio(r_m, weights)

ff = load_ff5_mom_monthly(config.FF5MOM_FILE)
model = align_factors(port_rets[["Net"]], ff)

# Extract metrics that actually work
alpha_ann = (1 + model.params['const'])**12 - 1 if 'const' in model.params else np.nan
alpha_tstat = model.tvalues.get('const', np.nan)

# Print metrics for parsing
print(f"ALPHA_ANN: {{alpha_ann}}")
print(f"ALPHA_TSTAT: {{alpha_tstat}}")
"""
        
        # Write temp script
        with open('_temp_backtest_metrics.py', 'w') as f:
            f.write(temp_script)
        
        # Run in subprocess (suppress main.py's graphs/output)
        result = subprocess.run(
            ['python', '_temp_backtest_metrics.py'],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        # Parse output to extract metrics
        output = result.stdout + result.stderr
        metrics = {
            'ridge_alpha': ridge_alpha,
            'alpha_ann': np.nan,
            'alpha_tstat': np.nan,
            'status': 'FAILED'
        }
        
        # Parse metrics from output
        for line in output.split('\n'):
            if 'ALPHA_ANN:' in line:
                try:
                    metrics['alpha_ann'] = float(line.split('ALPHA_ANN:')[1].strip())
                except:
                    pass
            elif 'ALPHA_TSTAT:' in line:
                try:
                    metrics['alpha_tstat'] = float(line.split('ALPHA_TSTAT:')[1].strip())
                except:
                    pass
        
        if result.returncode == 0 and not np.isnan(metrics['alpha_ann']):
            metrics['status'] = 'SUCCESS'
            print(f"✓ α={metrics['alpha_ann']:.4f}, t={metrics['alpha_tstat']:.2f}")
        else:
            print(f"✗ Failed")
            if result.returncode != 0:
                metrics['error'] = f"Exit code: {result.returncode}"
        
        # Clean up temp script
        if os.path.exists('_temp_backtest_metrics.py'):
            os.remove('_temp_backtest_metrics.py')
        
        return metrics
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return {
            'ridge_alpha': ridge_alpha,
            'status': 'FAILED',
            'error': str(e)[:100]
        }


def main():
    print("\n" + "="*80)
    print("RIDGE ALPHA GRID SEARCH")
    print("="*80)
    print(f"Testing alphas: {RIDGE_ALPHAS_TO_TEST}")
    print()
    
    results = []
    
    for i, alpha in enumerate(RIDGE_ALPHAS_TO_TEST, 1):
        print(f"[{i:2d}/{len(RIDGE_ALPHAS_TO_TEST)}]", end=" ")
        metrics = run_single_alpha_backtest(alpha)
        results.append(metrics)
        gc.collect()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
     
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Filter successful runs
    successful = results_df[results_df['status'] == 'SUCCESS'].copy()
    if len(successful) > 0:
        # Sort by alpha_tstat (significance)
        successful = successful.sort_values('alpha_tstat', ascending=False)
        
        # Display metrics - only what works
        display_cols = ['ridge_alpha', 'alpha_ann', 'alpha_tstat']
        display_df = successful[display_cols].copy()
        
        # Format for better readability
        print("\n" + display_df.to_string(index=False))
        
        # Summary statistics
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        best_alpha_idx = successful['alpha_ann'].idxmax()
        best_tstat_idx = successful['alpha_tstat'].idxmax()
        
        print(f"Best Alpha:           α={successful.loc[best_alpha_idx, 'ridge_alpha']:.1f} "
              f"→ {successful.loc[best_alpha_idx, 'alpha_ann']:.4f} "
              f"(t={successful.loc[best_alpha_idx, 'alpha_tstat']:.2f})")
        
        print(f"Most Significant:     α={successful.loc[best_tstat_idx, 'ridge_alpha']:.1f} "
              f"→ {successful.loc[best_tstat_idx, 'alpha_ann']:.4f} "
              f"(t={successful.loc[best_tstat_idx, 'alpha_tstat']:.2f})")
        
        print(f"\nResults saved to: {csv_path}")
    
    # Show failures
    failed = results_df[results_df['status'] == 'FAILED']
    if len(failed) > 0:
        print(f"\n  Failed runs: {len(failed)}")
        for _, row in failed.iterrows():
            print(f"   α={row['ridge_alpha']}: {row.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
