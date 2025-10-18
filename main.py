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
    #rel_mom_daily = load_relative_momentum_panel()
    sigs_raw = load_signals()

    # 2) ETF monthly returns
    r_m = etf_monthly_returns(etf_px)
    r_m = r_m.loc[BACKTEST_START:BACKTEST_END]
    r_m = apply_inception_mask(r_m, ETF_INCEPTION)

    # 3) Monthly signals (lag 1M) + z-scores
    sig_m = monthly_signal_panel(sigs_raw)
    sig_m = sig_m.loc[BACKTEST_START:BACKTEST_END]
    z = zscore_signals(sig_m)

    # 4) Sector-specific rolling betas
    betas = rolling_sectors_betas(r_m, z, lookback_months=36, min_months=24)

    # 5) Per-sector forecasts (predict t+1 using z_t)
    fcast = sector_expected_returns(betas, z)

    # 6) Vol for inverse-vol option
    vol_m = monthly_volatility(r_m, LOOKBACK_VOL_MONTHS)

    # 7) Build portfolio from forecasts
    weights = rank_and_weight_from_forecast(fcast, r_m, vol_m)

    # 8) Simulate (weights at t applied to r_{t+1})
    port_rets = simulate_portfolio(r_m, weights)

    # 9) FF5 + Momentum alpha
    ff = load_ff5_mom_monthly(FF5MOM_FILE)
    print(port_rets.head())
    print(port_rets.columns)
    model = align_factors(port_rets["Net"].to_frame("Net"), ff)
    alpha_stats = describe_alpha(model)

    # 10) Performance summary
    summ = summary_table(port_rets, rf_series=ff[["RF"]])
    print("\n==== Portfolio Performance (monthly) ====\n")
    print(summ.round(4))
    print("\n==== FF5 + Momentum Alpha (HAC t-stats) ====\n")
    print(model.summary())
    print("\nAlpha Stats:",
          {k: (round(v, 4) if isinstance(v, (int, float)) and v is not None else v) for k, v in alpha_stats.items()})

    alpha_ann = (1 + model.params['const'])**12 - 1
    alpha_tstat = model.tvalues['const']
    print(f"Strategy alpha (annualized): {alpha_ann:.2%} (t={alpha_tstat:.2f})")

    print("\n==== EXTENDED METRICS ====\n")
    extra = extended_metrics(port_rets["Net"], ff_model=model)
    print(extra.apply(lambda x: round(x, 4)))

    # Plot cumulative performance
    plot_equity_curves(port_rets[["Gross", "Net"]])
    plot_drawdown(port_rets["Net"])
    plot_rolling_sharpe(port_rets["Net"], window=24)

    # Rolling alpha vs FF5+Mom
    plot_rolling_alpha(port_rets.rename(columns={"Net": "net"}), ff, window=24)

    # Factor exposures (from last regression)
    plot_factor_exposures(model)

    # Optional: visualize correlation between standardized signals
    plot_signal_correlation(z)


if __name__ == "__main__":
    main()
