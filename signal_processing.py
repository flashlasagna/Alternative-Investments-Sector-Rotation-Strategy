import numpy as np
import pandas as pd
from config import Z_WINDOW_MONTHS

def resample_to_month_end(df, how="last"):
    if how == "last":
        return df.resample("ME").last()
    elif how == "mean":
        return df.resample("ME").mean()
    elif how == "sum":
        return df.resample("ME").sum()
    else:
        raise ValueError("how must be 'last'|'mean'|'sum'")

def make_copper_gold_ratio(sig_dict):
    if "COPPER" in sig_dict and "GOLD" in sig_dict:
       cg = (sig_dict["COPPER"]/sig_dict["GOLD"]).dropna()
       cg.columns = ["CG_RATIO"]
       sig_dict["CG_RATIO"] = cg
    return sig_dict

def monthly_signal_panel(sig_dict):
    monthly =[]
    for name, df in sig_dict.items():
        df = df.sort_index()
        #dectect monthly vs. daily heuristic
        md = df.index.to_series().diff().median()
        if pd.notna(md) and md <= pd.Timedelta(days=7):
            mdf = resample_to_month_end(df, how="last")
        else:
            mdf = df.copy()
            mdf.index = mdf.index + pd.offsets.MonthEnd(0)
        mdf = mdf.rename(columns={mdf.columns[0]: name})
        monthly.append(mdf)
    panel = pd.concat(monthly, axis=1).sort_index()
    panel = panel.shift(1)  # lag by 1 month to avoid lookahead bias
    return panel

def zscore_signals(panel, window=Z_WINDOW_MONTHS, min_periods=12):
    z = (panel-panel.rolling(window, min_periods=min_periods).mean()) / panel.rolling(window, min_periods=min_periods).std()
    return z

def etf_monthly_returns(etf_prices_dict):
    rets =[]
    for tkr, df in etf_prices_dict.items():
        m = df["AdjClose"].resample("ME").last().pct_change()
        rets.append(m.rename(tkr))
    return pd.concat(rets, axis=1).sort_index()

def apply_inception_mask(ret_df, inception_map):
    for tkr, d in inception_map.items():
       d0 = pd.to_datetime(d)+ pd.offsets.MonthEnd(0)
       ret_df.loc[: d0 - pd.offsets.MonthBegin(1), tkr] = np.nan
    return ret_df

def monthly_volatility(ret_df, lookback_months=6):
    return ret_df.rolling(lookback_months).std()