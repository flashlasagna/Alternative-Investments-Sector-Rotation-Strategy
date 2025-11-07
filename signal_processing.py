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


def yield_curve_dummy(panel_monthly):
    """
    Add yield curve regime dummies to monthly panel.
    - YC_EXPANSION = 1 if (10Y - 2Y) > 0 (steep/expansion), else 0.
    - YC_CONTRACTION = 1 if (10Y - 2Y) <= 0 (flat/inverted/contraction), else 0.
    Uses either precomputed spread 'Y10_2Y' or difference of 'YIELD_10Y'-'YIELD_2Y'.
    """
    panel = panel_monthly.copy()
    if 'Y10_2Y' in panel.columns:
        yc_spread = panel['Y10_2Y']
    elif 'T10Y2Y' in panel.columns:
        yc_spread = panel['T10Y2Y']
    elif {'YIELD_10Y', 'YIELD_2Y'} <= set(panel.columns):
        yc_spread = panel['YIELD_10Y'] - panel['YIELD_2Y']
    else:
        return panel_monthly
    panel['YC_EXPANSION'] = (yc_spread > 0).astype(int)
    panel['YC_CONTRACTION'] = (yc_spread <= 0).astype(int)

    # Additional slope regime dummies using intuitive thresholds (in percentage points)
    panel['YC_DEEP_INVERSION'] = (yc_spread <= -0.50).astype(int)
    panel['YC_MILD_INVERSION'] = ((yc_spread > -0.50) & (yc_spread <= 0)).astype(int)
    panel['YC_FLAT'] = ((yc_spread > 0) & (yc_spread <= 0.50)).astype(int)
    panel['YC_STEEP'] = (yc_spread > 0.50).astype(int)

    # Ensure mutually exclusive regimes by zeroing rows that do not belong
    regimes = ['YC_DEEP_INVERSION', 'YC_MILD_INVERSION', 'YC_FLAT', 'YC_STEEP']
    regime_sum = panel[regimes].sum(axis=1)
    for col in regimes:
        panel.loc[regime_sum == 0, col] = 0
    return panel


def business_cycle_dummy(panel_monthly, nber_recession_series):
    """
    Add business cycle (NBER-style) dummies to monthly panel.
    - BC_EXPANSION = 1 if not recession.
    - BC_RECESSION = 1 if recession.
    Args:
        panel_monthly (DataFrame).
        nber_recession_series (pd.Series): 1 = recession, 0 = expansion.
    """
    panel = panel_monthly.copy()
    if isinstance(nber_recession_series, pd.DataFrame):
        nber_recession_series = nber_recession_series.squeeze()
    aligned = nber_recession_series.reindex(panel.index).fillna(0).astype(int)
    panel['BC_RECESSION'] = aligned
    panel['BC_EXPANSION'] = (1 - panel['BC_RECESSION']).astype(int)

    # Lead/Lag recession awareness (3-month window)
    panel['BC_RECESSION_LEAD3'] = aligned.shift(-3).fillna(0).astype(int)
    panel['BC_RECESSION_LAG3'] = aligned.shift(3).fillna(0).astype(int)

    # Recovery indicator: first six months after recession ends
    recession_end = (aligned.diff() == -1)
    recovery = recession_end.shift(1, fill_value=False)
    recovery = recovery.astype(int).rolling(window=6, min_periods=1).max()
    panel['BC_RECOVERY'] = recovery.reindex(panel.index).fillna(0).astype(int)
    return panel


def monthly_signal_panel(sig_dict, nber_recession_series=None):
    """
    1) Build base monthly series for each raw signal (daily -> month-end last; monthly -> month-end).
    2) Create engineered features (LOG_*, D1_*, D3_*).
    3) Apply availability lag: shift(1) on the WHOLE expanded panel.
    4) Clean index (month-end, sorted, deduped).
    5) Add regime dummies if requested.
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
    # Allow raw spread column name straight from FRED (T10Y2Y)
    if 'T10Y2Y' in base.columns and 'Y10_2Y' not in base.columns:
        base = base.rename(columns={'T10Y2Y': 'Y10_2Y'})

    # 2) Add engineered transforms on BASE (no lag yet)
    expanded = _add_transforms(base)

    # 3) Apply 1-month availability lag to EVERYTHING
    expanded = expanded.shift(1)

    # 4) Normalize to month-end and clean index
    expanded.index = expanded.index + pd.offsets.MonthEnd(0)
    expanded = expanded[~expanded.index.duplicated(keep="last")].sort_index()

    # 5) Drop all-NaN columns (e.g., logs failing positivity)
    expanded = expanded.dropna(axis=1, how="all")

    # 6) Add regime dummies if appropriate
    # Yield curve: support either Y10_2Y or component yields
    if ('Y10_2Y' in expanded.columns) or (
        'YIELD_10Y' in expanded.columns and 'YIELD_2Y' in expanded.columns
    ):
        expanded = yield_curve_dummy(expanded)
    if nber_recession_series is not None:
        expanded = business_cycle_dummy(expanded, nber_recession_series)

    return expanded


def zscore_signals(panel, window=Z_WINDOW_MONTHS, min_periods=12):
    roll_mean = panel.rolling(window, min_periods=min_periods).mean()
    roll_std = panel.rolling(window, min_periods=min_periods).std()
    z = (panel - roll_mean) / roll_std

    # Identify binary/dummy columns (values only 0/1/NaN over history) and skip z-scoring
    def _is_binary(s):
        vals = s.dropna().unique()
        return len(vals) <= 2 and set(np.round(vals, 0)).issubset({0, 1})

    binary_cols = [c for c in panel.columns if _is_binary(panel[c])]
    if binary_cols:
        z[binary_cols] = panel[binary_cols]

    z = z.replace([np.inf, -np.inf], np.nan)
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