import numpy as np
import pandas as pd
from config import LONG_SHORT_PAIRS, USE_VOL_TARGETING

def rank_and_weight_from_forecast(forecast_df, monthly_rets, vol_df):
    """
    Rank by forecast per month; long top N, short bottom N.
    Optionally inverse-vol scale inside long and short buckets.
    Returns a DataFrame of monthly weights.
    """
    weights = []
    for t in forecast_df.index:
        f = forecast_df.loc[t].dropna()
        if f.empty:
            weights.append(pd.Series(dtype=float, name=t))
            continue

        avail = monthly_rets.columns[monthly_rets.loc[t].notna()]
        f = f.reindex(avail).dropna()
        if f.empty:
            weights.append(pd.Series(dtype=float, name=t))
            continue

        f = f.sort_values(ascending=False)
        n = min(LONG_SHORT_PAIRS, len(f) // 2)
        if n < 1:
            weights.append(pd.Series(dtype=float, name=t))
            continue

        longs = f.index[:n]
        shorts = f.index[-n:]

        w = pd.Series(0.0, index=avail, name=t)
        if not USE_VOL_TARGETING or t not in vol_df.index:
            w.loc[longs] = 1.0 / n
            w.loc[shorts] = -1.0 / n
        else:
            vols = vol_df.loc[t].reindex(longs.union(shorts)).replace(0, np.nan)
            inv = 1.0 / vols
            invL = inv.reindex(longs).dropna()
            invS = inv.reindex(shorts).dropna()
            if not invL.empty:
                w.loc[invL.index] = invL / invL.sum()
            if not invS.empty:
                w.loc[invS.index] = -invS / invS.sum()
        weights.append(w)
    W = pd.DataFrame(weights).sort_index().fillna(0.0)
    return W

def scale_to_target_vol(portfolio_returns, weights, vol_target=0.12, lookback_months=12, smoothing_window=3):
    """
    Scales portfolio weights for target annualized volatility, then smooths scaling factors to avoid jumps.
    """
    scaling_factors = []
    for t in weights.index:
        if t not in portfolio_returns.index:
            scaling_factors.append(1.0)
            continue
        rets_window = portfolio_returns.loc[:t].tail(lookback_months)
        port_rets = rets_window.dot(weights.loc[t])
        realized_vol = port_rets.std() * np.sqrt(12)
        scaling = vol_target / realized_vol if realized_vol > 0 else 1.0
        scaling_factors.append(scaling)

    # Smoothing: rolling mean on scaling factors (centered, min_periods=1)
    scaling_series = pd.Series(scaling_factors, index=weights.index)
    scaling_smoothed = scaling_series.rolling(window=smoothing_window, min_periods=1, center=True).mean()

    # Apply smoothed scaling to monthly weights
    scaled_weights = weights.copy()
    for t in weights.index:
        scaled_weights.loc[t] *= scaling_smoothed.loc[t]

    return scaled_weights

# Example Usage:
# 1. Generate raw weights by forecast
# raw_weights = rank_and_weight_from_forecast(forecast_df, monthly_rets, vol_df)
# 2. Scale final weights to hit overall volatility target (e.g. 10% annualized)
# final_weights = scale_to_target_vol(monthly_rets, raw_weights, vol_target=0.10, lookback_months=6)

