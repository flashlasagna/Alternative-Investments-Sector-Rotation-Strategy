import numpy as np
import pandas as pd
from config import LONG_SHORT_PAIRS, USE_VOL_TARGETING

def rank_and_weight_from_forecast(forecast_df, monthly_rets, vol_df):
    """
        Rank by forecast per month; long top N, short bottom N.
        Optionally inverse-vol scale inside long and short buckets.
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
