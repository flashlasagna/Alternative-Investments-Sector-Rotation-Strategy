import numpy as np
import pandas as pd
from config import Z_WINDOW_MONTHS, LOG_COLS, DIFF1_COLS, DIFF3_COLS, ADD_LOG_CG


def _safe_log(s: pd.Series) -> pd.Series:
    """Log transform only where values are strictly positive; NaN otherwise."""
    out = s.copy()
    out[~(out > 0)] = np.nan
    return np.log(out)


def _add_transforms(panel_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Expand monthly panel with engineered features:
      - LOG_*
      - D1_* (1-month difference)
      - D3_* (3-month difference)
    Assumes input is monthly and NOT yet availability-lagged.
    """
    out = panel_monthly.copy()

    # 1) Log transforms
    for col in [c for c in LOG_COLS if c in out.columns]:
        out[f"LOG_{col}"] = _safe_log(out[col])

    # If CG ratio exists and flag set, also ensure its log is present
    if ADD_LOG_CG and "CG_RATIO" in out.columns and "LOG_CG_RATIO" not in out.columns:
        out["LOG_CG_RATIO"] = _safe_log(out["CG_RATIO"])

    # 2) 1-month differences
    for col in [c for c in DIFF1_COLS if c in out.columns]:
        out[f"D1_{col}"] = out[col].diff(1)

    # 3) 3-month differences (captures acceleration)
    for col in [c for c in DIFF3_COLS if c in out.columns]:
        out[f"D3_{col}"] = out[col].diff(3)

    # Optional: percent changes (uncomment if you want)
    # for col in ["OIL","DXY","VIX","COPPER","GOLD"]:
    #     if col in out.columns:
    #         out[f"ROC1_{col}"] = out[col].pct_change(1)

    return out


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
        # Convert single-column DataFrames to Series to avoid column alignment issues
        copper_series = sig_dict["COPPER"].squeeze()
        gold_series = sig_dict["GOLD"].squeeze()
        
        # Divide and convert back to DataFrame with proper column name
        cg = (copper_series / gold_series).dropna().to_frame(name="CG_RATIO")
        sig_dict["CG_RATIO"] = cg
    return sig_dict


def monthly_signal_panel(sig_dict):
    """
    1) Build base monthly series for each raw signal (daily -> month-end last; monthly -> month-end).
    2) Create engineered features (LOG_*, D1_*, D3_*).
    3) Apply availability lag: shift(1) on the WHOLE expanded panel.
    4) Clean index (month-end, sorted, deduped).
    """
    sig_dict = make_copper_gold_ratio(sig_dict)

    monthly = []
    for name, df in sig_dict.items():
        df = df.sort_index()
        # detect monthly vs. daily heuristic
        md = df.index.to_series().diff().median()
        if pd.notna(md) and getattr(md, "days", 9999) <= 7:
            mdf = resample_to_month_end(df, how="last")
        else:
            mdf = df.copy()
            mdf.index = mdf.index + pd.offsets.MonthEnd(0)
        mdf = mdf.rename(columns={mdf.columns[0]: name})
        monthly.append(mdf)

    base = pd.concat(monthly, axis=1).sort_index()

    # 2) Add engineered transforms on BASE (no lag yet)
    expanded = _add_transforms(base)

    # 3) Apply 1-month availability lag to EVERYTHING
    expanded = expanded.shift(1)

    # 4) Normalize to month-end and clean index
    expanded.index = expanded.index + pd.offsets.MonthEnd(0)
    expanded = expanded[~expanded.index.duplicated(keep="last")].sort_index()

    # 5) Drop all-NaN columns (e.g., logs failing positivity)
    expanded = expanded.dropna(axis=1, how="all")

    return expanded


def zscore_signals(panel, window=Z_WINDOW_MONTHS, min_periods=12):
    z = (panel - panel.rolling(window, min_periods=min_periods).mean()) / panel.rolling(window, min_periods=min_periods).std()
    return z


def etf_monthly_returns(etf_prices_dict):
    rets = []
    for tkr, df in etf_prices_dict.items():
        m = df["AdjClose"].resample("ME").last().pct_change()
        rets.append(m.rename(tkr))
    return pd.concat(rets, axis=1).sort_index()


def apply_inception_mask(ret_df, inception_map):
    for tkr, d in inception_map.items():
        d0 = pd.to_datetime(d) + pd.offsets.MonthEnd(0)
        ret_df.loc[: d0 - pd.offsets.MonthBegin(1), tkr] = np.nan
    return ret_df


def monthly_volatility(ret_df, lookback_months=6):
    return ret_df.rolling(lookback_months).std()