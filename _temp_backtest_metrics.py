
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# Update config BEFORE importing other modules
import config
config.RIDGE_ALPHA = 50.0

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
print(f"ALPHA_ANN: {alpha_ann}")
print(f"ALPHA_TSTAT: {alpha_tstat}")
