from config import ( BACKTEST_START, BACKTEST_END, ETF_INCEPTION, LOOKBACK_VOL_MONTHS, FF5MOM_FILE, LONG_SHORT_PAIRS, USE_VOL_TARGETING, LOOKBACK_BETA_MONTHS, MIN_BETA_MONTHS,
                    CRISIS_VIX_PERCENTILE, CRISIS_CS_PERCENTILE, CRISIS_WINDOW_MONTHS, CRISIS_LOGIC, CRISIS_MIN_ON, CRISIS_MIN_OFF, CRISIS_REDUCTION)
from data_loader import (load_sector_etf_prices, load_signals, load_relative_momentum_signals, load_ff5_mom_monthly, load_vix_monthly, load_credit_spread_monthly)
from signal_processing import (etf_monthly_returns, apply_inception_mask, monthly_signal_panel, zscore_signals,
                               monthly_volatility)
from exposure_model import rolling_sectors_betas, sector_expected_returns
from portfolio_construction import rank_and_weight_from_forecast, rank_and_weight_from_forecast_hysteresis, apply_position_caps_and_renormalize, scale_to_target_vol
from crisis_filter import compute_crisis_flag, apply_crisis_reduction, get_crisis_info
from backtest import simulate_portfolio
from benchmark import align_factors, describe_alpha
from utils.performance import summary_table, extended_metrics
from utils.diagnostics import (plot_equity_curves, plot_drawdown, plot_rolling_sharpe, plot_rolling_alpha,
                               plot_factor_exposures, plot_signal_correlation, plot_cs_ic, plot_top_bottom_spread, plot_vol_scaling_series,
                               plot_sector_weights, plot_turnover, plot_subperiod_sharpe, plot_return_distribution)
def main():
    # 1) Load data
    etf_px = load_sector_etf_prices()
    sigs_raw = load_signals()
    
    # 1b) Load VIX and credit spread for crisis detection
    vix_monthly = load_vix_monthly()
    credit_spread_monthly = load_credit_spread_monthly()

    # 2) ETF monthly returns — keep FULL history for warm-up (no slicing yet)
    r_full = etf_monthly_returns(etf_px)
    r_full = apply_inception_mask(r_full, ETF_INCEPTION)
    
    # 2b) Compute crisis flag (FULL HISTORY)
    crisis_flag_full = compute_crisis_flag(
        vix=vix_monthly,
        credit_spread=credit_spread_monthly,
        window_months=CRISIS_WINDOW_MONTHS,
        p_vix=CRISIS_VIX_PERCENTILE,
        p_cs=CRISIS_CS_PERCENTILE,
        logic=CRISIS_LOGIC,
        min_on=CRISIS_MIN_ON,
        min_off=CRISIS_MIN_OFF
    )
    
    # Print crisis diagnostics
    crisis_info = get_crisis_info(crisis_flag_full)
    print("\n==== CRISIS PERIODS DETECTED ====")
    print(f"Total crisis months: {crisis_info['crisis_months']} / {crisis_info['total_months']} ({crisis_info['crisis_pct']:.1f}%)")
    if crisis_info['first_crisis'] is not None:
        print(f"First crisis: {crisis_info['first_crisis'].strftime('%Y-%m')}")
        print(f"Last crisis: {crisis_info['last_crisis'].strftime('%Y-%m')}")

    # 3) Monthly signals (lag 1M) + z-scores — keep FULL history for warm-up
    sig_full = monthly_signal_panel(sigs_raw)
    z_full = zscore_signals(sig_full)

    # 4) Sector-specific rolling betas on FULL history
    betas = rolling_sectors_betas(r_full, z_full, lookback_months=LOOKBACK_BETA_MONTHS, min_months=MIN_BETA_MONTHS)

    # 5) Per-sector forecasts (predict t+1 using z_t) on FULL history
    fcast_full = sector_expected_returns(betas, z_full)

    # 6) Vol for inverse-vol option (FULL history)
    vol_full = monthly_volatility(r_full, LOOKBACK_VOL_MONTHS)

    # 7) Build portfolio from forecasts (FULL history) with HYSTERESIS
    # Returns raw equal-weight buckets (scaled ±1.0), before caps/renorm
    weights_full, hysteresis_state = rank_and_weight_from_forecast_hysteresis(
        fcast_full, r_full, 
        entry_long=4, 
        exit_long=6,
        dwell_min=1,              # 1-month friction (can increase to 3 for more stability)
        min_bucket_size=3         # Maintain at least 3 longs / 3 shorts for neutrality
    )

    # 7b) Apply position caps and renormalize to target (market-neutral: 20L/20S)
    weights_full = apply_position_caps_and_renormalize(
        weights_full,
        L_target=0.20,            # 20% gross long
        S_target=0.20,            # 20% gross short (neutral)
        max_long_single=0.25,     # Max 25% per sector long
        max_short_single=0.15     # Max 15% per sector short
    )

    # 7c) Apply crisis reduction (PHASE 2)
    weights_full = apply_crisis_reduction(
        weights=weights_full,
        crisis_flag=crisis_flag_full,
        reduction=CRISIS_REDUCTION
    )

    # 7d) Portfolio-level volatility targeting (with crisis cap)
    weights_full = scale_to_target_vol(
        sector_returns=r_full, 
        weights=weights_full, 
        vol_target=0.12, 
        lookback_months=12, 
        smoothing_window=4,
        crisis_flag=crisis_flag_full  # Pass crisis flag to cap scaler!
    )

    # 8) Slice to backtest window ONLY after warm-up
    r_m = r_full.loc[BACKTEST_START:BACKTEST_END]
    weights = weights_full.loc[BACKTEST_START:BACKTEST_END]

    # 9) Simulate (weights at t applied to r_{t+1})
    port_rets = simulate_portfolio(r_m, weights)

    # 10) FF5 + Momentum alpha
    ff = load_ff5_mom_monthly(FF5MOM_FILE)

    # Align factors expects a 'net' column; rename defensively
    model = align_factors(port_rets[["Net"]], ff)
    alpha_stats = describe_alpha(model)

    # 11) Performance summary
    summ = summary_table(port_rets, rf_series=ff[["RF"]])
    print("\n==== Portfolio Performance (monthly) ====\n")
    print(summ.round(4))

    print("\n==== FF5 + Momentum Alpha (HAC t-stats) ====\n")
    print(model.summary())

    print("\nAlpha Stats:",
          {k: (round(v, 4) if isinstance(v, (int, float)) and v is not None else v)
           for k, v in alpha_stats.items()})

    alpha_ann = (1 + model.params['const'])**12 - 1 if 'const' in model.params else float('nan')
    alpha_tstat = model.tvalues.get('const', float('nan'))
    print(f"Strategy alpha (annualized): {alpha_ann:.2%} (t={alpha_tstat:.2f})")

    # 12) Extended metrics
    print("\n==== EXTENDED METRICS ====\n")
    extra = extended_metrics(port_rets["Net"], ff_model=model)
    print(extra.apply(lambda x: round(x, 4)))

    # 13) Plots
    plot_equity_curves(port_rets[["Gross", "Net"]])
    plot_drawdown(port_rets["Net"])
    plot_rolling_sharpe(port_rets["Net"], window=24)
    plot_rolling_alpha(port_rets.rename(columns={"Net": "net"}), ff, window=24)
    plot_factor_exposures(model)
    plot_signal_correlation(z_full)

    # 14) Diagnostics: first valid dates at each stage
    def first_valid(name, df_like):
        try:
            print(f"{name}: first non-NaN = {df_like.dropna(how='all').index.min()}")
        except Exception:
            print(f"{name}: n/a")

    first_valid("Signals (monthly, lagged)", sig_full)
    first_valid("Z-scores (FULL)", z_full)
    for s, bdf in betas.items():
        if isinstance(bdf, type(r_full)) and not bdf.empty:
            print(f"Betas[{s}]: first non-NaN = {bdf.index.min()}")
            break
    first_valid("Forecasts (FULL)", fcast_full)
    first_valid("Weights (FULL)", weights_full)
    first_valid("Weights (windowed)", weights)
    first_valid("Portfolio returns", port_rets)

    # Trading window diagnostics
    def first_last_trading(df):
        active = (df != 0).any(axis=1)
        active_dates = df.index[active]
        if len(active_dates) == 0:
            print(" No trading activity detected.")
            return
        first_trade = active_dates.min()
        last_trade = active_dates.max()
        print(f"\n First trading month: {first_trade.strftime('%Y-%m')}")
        print(f"Last trading month:  {last_trade.strftime('%Y-%m')}")
        print(f"Total active months: {len(active_dates)}")

    print("\n==== TRADING WINDOW CHECK ====")
    first_last_trading(weights)
    first_last_trading(port_rets)
    # Use FULL history for forecast diagnostics
    plot_cs_ic(fcast_full, r_full, roll=12)
    plot_top_bottom_spread(fcast_full, r_full, n=3)

    # Show vol-targeting behavior (uses final weights windowed; you can also use weights_full)
    plot_vol_scaling_series(r_m, weights, lookback=12, vol_target=0.12, smooth=4)

    # Weight dynamics + turnover (windowed)
    plot_sector_weights(weights, top_k=None)   # or top_k=8
    plot_turnover(weights)

    # Subperiod Sharpe & return distribution (Net)
    plot_subperiod_sharpe(port_rets["Net"], cuts=("2009-01","2015-01","2020-01","2024-12"))
    plot_return_distribution(port_rets["Net"])
if __name__ == "__main__":
    main()