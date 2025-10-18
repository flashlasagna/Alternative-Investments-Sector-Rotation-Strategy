import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
from statsmodels.api import OLS, add_constant

def rolling_sectors_betas(monthly_rets, z_signals, lookback_months = 36, min_months = 24):
    """
       For each sector i, run rolling time-series OLS:
           r_{i,t} = alpha_i + sum_k beta_{i,k}(t) * z_{k,t-1} + eps
       Using the past `lookback_months` months. Betas at t forecast r_{i,t+1}.
       Returns dict[sector] -> DataFrame(index=t, columns=['const', signals...]).
    """
    common_idx = monthly_rets.index.intersection(z_signals.index)
    r = monthly_rets.loc[common_idx]
    z = z_signals.loc[common_idx]
    X_lag = z.shift(1)

    betas = {}
    for sector in r.columns:
        y = r[sector]
        df = pd.concat([y, X_lag], axis=1).dropna(how="any")
        if df.empty:
            betas[sector] = pd.DataFrame()
            continue

        rows =[]
        idx = df.index
        for i in range(len(idx)):
            end_pos = i
            start_pos = i -(lookback_months - 1)
            if start_pos < 0:
                continue
            window = idx[start_pos:end_pos + 1]
            if len(window) < min_months:
                continue
            sub = df.loc[window]
            y_win = sub.iloc[:,0]
            X_win = sub.iloc[:,1:]
            X_win = add_constant(X_win)
            try:
                model = OLS(y_win.values, X_win.values).fit()
                params = pd.Series(model.params, index=X_win.columns, name=window[-1])
                rows.append(params)
            except Exception:
                continue
            betas_df = pd.DataFrame(rows).sort_index()
        betas[sector] = betas_df
    return betas

def sector_expected_returns(betas_dict, z_signals):
    idx = z_signals.index
    sectors = list(betas_dict.keys())
    out = pd.DataFrame(index=idx, columns=sectors, dtype=float)

    for sector, bdf in betas_dict.items():
        if bdf.empty:
            continue
        common_t = bdf.index.intersection(idx)
        for t in common_t:
            rowb = bdf.loc[t]
            c = rowb.get('const', 0.0)
            sig_cols = [cname for cname in rowb.index if cname != 'const' and cname in z_signals.columns]
            zt = z_signals.loc[t, sig_cols]
            out.loc[t, sector] = c + float((rowb[sig_cols] * zt).sum())
    return out

