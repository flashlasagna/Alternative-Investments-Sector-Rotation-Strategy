from config import (BACKTEST_START, BACKTEST_END, ETF_INCEPTION, LOOKBACK_VOL_MONTHS, FF5MOM_FILE)
from data_loader import (load_sector_etf_prices, load_signals, load_relative_momentum_signals, load_ff5_mom_monthly)
from signal_processing import (etf_monthly_returns, apply_inception_mask, monthly_signal_panel, zscore_signals, monthly_volatility)
from exposure_model import rolling_sectors_betas, sector_expected_returns
from portfolio_construction import rank_and_weight_from_forecast
from backtest import simulate_portfolio
from benchmark import  align_factors, describe_alpha
from utils.performance import summary_table, extended_metrics
from utils.plotting import plot_equity_curves, plot_drawdown, plot_rolling_sharpe
from utils.diagnostics import (plot_equity_curves, plot_drawdown, plot_rolling_sharpe, plot_rolling_alpha, plot_factor_exposures, plot_signal_correlation
)

def main():
    # 1) Load data
    etf_px = load_sector_etf_prices()
    sigs_raw = load_signals()

    # 2) ETF monthly returns — keep FULL history for warm-up (no slicing yet)
    r_full = etf_monthly_returns(etf_px)
    r_full = apply_inception_mask(r_full, ETF_INCEPTION)

    # 3) Monthly signals (lag 1M) + z-scores — keep FULL history for warm-up
    sig_full = monthly_signal_panel(sigs_raw)
    z_full = zscore_signals(sig_full)

    # 4) Sector-specific rolling betas on FULL history
    betas = rolling_sectors_betas(r_full, z_full, lookback_months=36, min_months=24)

    # 5) Per-sector forecasts (predict t+1 using z_t) on FULL history
    fcast_full = sector_expected_returns(betas, z_full)

    # 6) Vol for inverse-vol option (FULL history)
    vol_full = monthly_volatility(r_full, LOOKBACK_VOL_MONTHS)

    # 7) Build portfolio from forecasts (FULL history)
    weights_full = rank_and_weight_from_forecast(fcast_full, r_full, vol_full)

    # 8) Slice to backtest window ONLY after warm-up
    r_m = r_full.loc[BACKTEST_START:BACKTEST_END]
    weights = weights_full.loc[BACKTEST_START:BACKTEST_END]

    # 9) Simulate (weights at t applied to r_{t+1})
    port_rets = simulate_portfolio(r_m, weights)

    # 10) FF5 + Momentum alpha
    ff = load_ff5_mom_monthly(FF5MOM_FILE)

    # Quick peek
    print(port_rets.head())
    print(port_rets.columns)

    # Align factors expects a 'net' column; rename defensively
    net_for_ff = port_rets[["Net"]].rename(columns={"Net": "net"})
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

    # Rolling alpha vs FF5+Mom (expects 'net' col)
    plot_rolling_alpha(port_rets.rename(columns={"Net": "net"}), ff, window=24)

    # Factor exposures (from last regression)
    plot_factor_exposures(model)

    # Optional: visualize correlation between standardized signals (use FULL z panel)
    plot_signal_correlation(z_full)

    # 14) Diagnostics: first valid dates at each stage
    def first_valid(name, df_like):
        try:
            print(f"{name}: first non-NaN = {df_like.dropna(how='all').index.min()}")
        except Exception:
            print(f"{name}: n/a")

    first_valid("Signals (monthly, lagged)", sig_full)
    first_valid("Z-scores (FULL)", z_full)

    # first sector with betas:
    for s, bdf in betas.items():
        if isinstance(bdf, type(r_full)) and not bdf.empty:
            print(f"Betas[{s}]: first non-NaN = {bdf.index.min()}")
            break

    first_valid("Forecasts (FULL)", fcast_full)
    first_valid("Weights (FULL)", weights_full)
    first_valid("Weights (windowed)", weights)
    first_valid("Portfolio returns", port_rets)
    first_valid("Portfolio returns", port_rets)

    # === Detect first and last trading dates ===
    def first_last_trading(df):
        """Find first and last months where the portfolio is actually active."""
        # net/gross returns nonzero
        active = (df != 0).any(axis=1)
        active_dates = df.index[active]
        if len(active_dates) == 0:
            print("⚠️ No trading activity detected.")
            return
        first_trade = active_dates.min()
        last_trade = active_dates.max()
        print(f"\n🔹 First trading month: {first_trade.strftime('%Y-%m')}")
        print(f"🔹 Last trading month:  {last_trade.strftime('%Y-%m')}")

        # Optional: show how many months traded
        print(f"🔹 Total active months: {len(active_dates)}")

    # Check both weights and realized returns
    print("\n==== TRADING WINDOW CHECK ====")
    first_last_trading(weights)
    first_last_trading(port_rets)


if __name__ == "__main__":
    main()
