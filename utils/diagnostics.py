# # diagnostics.py
# import os
# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import statsmodels.api as sm

# FIG_DIR = "figs"
# os.makedirs(FIG_DIR, exist_ok=True)

# _CANON_FACTORS = ["Mkt_RF","SMB","HML","RMW","CMA","Mom","RF"]


# # -------------------------- SAVE HELPERS (PDF) -------------------------- #
# def _safe_name(s: str) -> str:
#     """Sanitize a string to be a safe filename."""
#     s = s.strip().lower()
#     s = re.sub(r"[^\w\-]+", "_", s)
#     s = re.sub(r"_+", "_", s).strip("_")
#     return s or "figure"

# def _savefig_pdf(name: str, title: str | None = None):
#     """
#     Save current matplotlib figure to FIG_DIR as PDF (tight layout).
#     File name is derived from `name` (and optional title).
#     """
#     fname = _safe_name(name if title is None else f"{name}_{title}")
#     path = os.path.join(FIG_DIR, f"{fname}.pdf")
#     plt.savefig(path, bbox_inches="tight")
#     # (If you ever want PNGs too, add: plt.savefig(path.replace('.pdf','.png'), dpi=300))


# # -------------------------- FACTOR STANDARDIZER -------------------------- #
# def _standardize_ff_columns(ff: pd.DataFrame) -> pd.DataFrame:
#     """Map various FF column variants to canonical names and fix percent/decimal scale."""
#     colmap = {}
#     for c in ff.columns:
#         cl = c.strip().lower()
#         if cl in ("mkt-rf","mkt_rf","mktrf","market-rf","market_rf","mkt_rf (%)","mkt minus rf","excess market return"):
#             colmap[c] = "Mkt_RF"
#         elif cl == "smb":
#             colmap[c] = "SMB"
#         elif cl == "hml":
#             colmap[c] = "HML"
#         elif cl == "rmw":
#             colmap[c] = "RMW"
#         elif cl == "cma":
#             colmap[c] = "CMA"
#         elif cl in ("mom","umd","momentum"):
#             colmap[c] = "Mom"
#         elif cl in ("rf","riskfree","risk_free","risk-free","risk free"):
#             colmap[c] = "RF"

#     ff2 = ff.rename(columns=colmap).copy()

#     # Convert percent -> decimal if needed (heuristic on numeric magnitude)
#     num = ff2.select_dtypes(include=[np.number])
#     if not num.empty and num.abs().mean().mean() > 0.5:
#         for c in num.columns:
#             ff2[c] = ff2[c] / 100.0

#     return ff2


# # -------------------------- CORE PLOTS (PDF) -------------------------- #
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import pandas as pd

# def plot_equity_curves(df, title="Strategy Cumulative Returns"):
#     # Ensure the index is datetime
#     if not pd.api.types.is_datetime64_any_dtype(df.index):
#         df.index = pd.to_datetime(df.index)
    
#     # Filter dates between 1 Jan 2009 and 31 Dec 2024
#     start_date = "2009-01-01"
#     end_date = "2024-12-31"
#     df_filtered = df.loc[start_date:end_date]
    
#     # Plot cumulative returns
#     ax = (1 + df_filtered.fillna(0)).cumprod().plot(title=title, lw=1.3)
#     plt.ylabel("Growth of $1")
#     plt.xlabel("Date")
#     plt.grid(True)
    
#     # Set x-axis ticks yearly from 2009 to 2024
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')  # YS = Year Start
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)  # Show only the year
    
#     plt.tight_layout()
#     _savefig_pdf("equity_curve", title)
#     plt.show()

# def plot_drawdown(r, title="Drawdown"):
    
#     start_date = "2010-01-01"
#     end_date = "2024-12-31"
#     r_filtered = r.loc[start_date:end_date]
    
#     eq = (1 + r_filtered).cumprod()
#     dd = eq / eq.cummax() - 1
    
#     ax = dd.plot(title=title, color="tomato", lw=1.3)
#     plt.ylabel("Drawdown")
#     plt.xlabel("Date")
    
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
    
#     plt.tight_layout()
#     _savefig_pdf("drawdown", title)
#     plt.show()


# def plot_rolling_sharpe(r, window=24):
#     title = f"Rolling Sharpe ({window}M)"
    
#     # Ensure the index is datetime
#     if not pd.api.types.is_datetime64_any_dtype(r.index):
#         r.index = pd.to_datetime(r.index)
    
#     # Compute rolling Sharpe on full series
#     roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(12)
    
#     # Filter only for plotting from 2010 onwards
#     start_date = "2011-01-01"
#     end_date = "2024-12-31"
#     roll_sharpe_plot = roll_sharpe.loc[start_date:end_date]
#     ax = roll_sharpe_plot.plot(title=title, lw=1.3, color="steelblue")
#     plt.axhline(0, color="gray", ls="--")
#     plt.ylabel("Sharpe Ratio")
#     plt.xlabel("Date")
#     # Set x-axis ticks yearly from 2011 to 2024
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
    
#     plt.tight_layout()
#     _savefig_pdf(f"rolling_sharpe_w{window}", title)
#     plt.show()


# def plot_rolling_alpha(port_df: pd.DataFrame, ff: pd.DataFrame, window: int = 24, title: str = None):
#     """
#     Rolling alpha (OLS, no HAC) vs. FF5 + Mom.
#     - port_df: must contain 'Net' or 'net'
#     - ff: FF factors DataFrame with any reasonable column naming; this function normalizes them
#     """
#     # Normalize portfolio column name
#     p = port_df.copy()
#     if "Net" in p.columns:
#         p = p.rename(columns={"Net": "net"})
#     elif "net" not in p.columns:
#         raise KeyError("Portfolio DataFrame must contain 'Net' or 'net' column.")

#     # Standardize factor names and scale
#     ff2 = _standardize_ff_columns(ff)

#     # Ensure we have at least RF and Market
#     needed_min = {"RF", "Mkt_RF"}
#     if not needed_min.issubset(set(ff2.columns)):
#         missing = needed_min - set(ff2.columns)
#         raise KeyError(f"Missing required factor column(s): {missing} after standardization.")

#     # Join and drop NA
#     aligned = p.join(ff2, how="inner").dropna(subset=["net", "RF", "Mkt_RF"])
#     aligned = aligned.sort_index()

#     # Choose available factor set (don’t error if some are missing)
#     factor_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in aligned.columns]

#     # Compute rolling alpha series
#     alphas = []
#     idx = aligned.index
#     for i in range(window, len(idx) + 1):
#         sub = aligned.iloc[i - window:i].copy()
#         y = sub["net"] - sub["RF"]
#         X = sm.add_constant(sub[factor_cols])
#         try:
#             mdl = sm.OLS(y, X).fit()
#             alphas.append(mdl.params.get("const", np.nan))
#         except Exception:
#             alphas.append(np.nan)

#     alpha_series = pd.Series(alphas, index=idx[window - 1:])
#     if title is None:
#         title = f"Rolling Alpha (Window={window} months)"

#   # Plot (annualized alpha)
#     ann_alpha = (1 + alpha_series) ** 12 - 1
#     # Filter plot to 2011-2024
#     ann_alpha_filtered = ann_alpha.loc['2011':'2024-12']
    
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ann_alpha_filtered.plot(ax=ax, title=title)
#     ax.axhline(0.0, ls="--", color="gray")
#     ax.set_ylabel("Annualized Alpha")
#     ax.set_xlabel("Date")
    
#     # Set x-axis to show every year
#     years = pd.date_range(start='2011', end='2024', freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45, ha='right')
    
#     plt.tight_layout()
#     _savefig_pdf(f"rolling_alpha_w{window}", title)
#     plt.show()

# def plot_factor_exposures(model):
#     """Bar chart of factor coefficients from latest regression."""
#     params = model.params.drop("const", errors="ignore")
#     params.plot(kind="bar", color="mediumpurple", title="Factor Exposures (FF5 + Momentum)")
#     plt.axhline(0, color="gray", ls="--")
#     plt.tight_layout()
#     _savefig_pdf("factor_exposures", "ff5_mom")
#     plt.show()


# def plot_signal_correlation(z_signals):
#     """Optional: visualize correlation among standardized signals."""
#     corr = z_signals.corr()
#     plt.figure(figsize=(9, 7))
#     sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
#     plt.title("Signal Correlation Matrix")
#     plt.tight_layout()
#     _savefig_pdf("signal_corr_heatmap")
#     plt.show()


# # -------------------------- UTILITIES -------------------------- #
# def first_last_trading(df):
#     """Find first and last months where the portfolio is actually active."""
#     active = (df != 0).any(axis=1)
#     active_dates = df.index[active]
#     if len(active_dates) == 0:
#         print("⚠️ No trading activity detected.")
#         return
#     first_trade = active_dates.min()
#     last_trade  = active_dates.max()
#     print(f"\n🔹 First trading month: {first_trade.strftime('%Y-%m')}")
#     print(f"🔹 Last trading month:  {last_trade.strftime('%Y-%m')}")
#     print(f"🔹 Total active months: {len(active_dates)}")


# # -------------------------- NEW DIAGNOSTICS (PDF) -------------------------- #
# def _cs_ic_series(fcast: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
#     """
#     Spearman rank IC: forecast at t vs realized sector return at t+1.
#     Returns a monthly time series.
#     """
#     idx = fcast.index.intersection(rets.index)
#     out = []
#     for t in idx[:-1]:
#         f = fcast.loc[t].dropna()
#         r_next = rets.loc[t + pd.offsets.MonthEnd(1)].reindex(f.index)
#         common = f.index.intersection(r_next.dropna().index)
#         if len(common) < 5:
#             out.append((t, np.nan))
#             continue
#         ic = f.loc[common].rank().corr(r_next.loc[common].rank(), method="spearman")
#         out.append((t, ic))
#     return pd.Series(dict(out)).sort_index()


# def plot_cs_ic(fcast: pd.DataFrame, rets: pd.DataFrame, roll=12, title=None):
#     """Cross-sectional IC with rolling mean."""
#     # Compute IC series on full data
#     ic = _cs_ic_series(fcast, rets)
#     start_date = "2011-01-01"
#     end_date = "2024-12-31"
#     ic_plot = ic.loc[start_date:end_date]
#     ax = ic_plot.plot(alpha=0.4, label="Monthly IC")
#     ic_plot.rolling(roll).mean().plot(ax=ax, linewidth=2, label=f"Rolling mean ({roll}m)")
#     plt.axhline(0, ls="--", color="gray")
#     m = ic.mean()
#     s = ic.std(ddof=1)
#     t = m / (s / np.sqrt(ic.dropna().shape[0])) if ic.dropna().shape[0] > 2 else np.nan
#     ttl = title or f"Cross-Sectional IC (mean={m:.3f}, t≈{t:.2f})"
#     plt.title(ttl)
#     plt.legend()
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
#     plt.tight_layout()
#     _savefig_pdf("cs_ic", ttl)
#     plt.show()



# def _tb_spread_series(fcast: pd.DataFrame, rets: pd.DataFrame, n=3) -> pd.Series:
#     """
#     Long top-n, short bottom-n by forecast at t; realized at t+1.
#     Returns a monthly L/S series.
#     """
#     idx = fcast.index.intersection(rets.index)
#     out = []
#     for t in idx[:-1]:
#         f = fcast.loc[t].dropna().sort_values(ascending=False)
#         if len(f) < 2*n:
#             out.append((t, np.nan)); continue
#         top = f.index[:n]; bot = f.index[-n:]
#         r_next = rets.loc[t + pd.offsets.MonthEnd(1)]
#         spread = r_next[top].mean() - r_next[bot].mean()
#         out.append((t, spread))
#     return pd.Series(dict(out)).sort_index()


# def plot_top_bottom_spread(fcast: pd.DataFrame, rets: pd.DataFrame, n=3, title=None):
#     s = _tb_spread_series(fcast, rets, n=n)
#     start_date = "2010-01-01"
#     end_date = "2024-12-31"
#     s_plot = s.loc[start_date:end_date]
#     ann = (1 + s.dropna().mean())**12 - 1 if s.dropna().size else np.nan
#     ax = (1 + s_plot.fillna(0)).cumprod().plot(lw=1.3)
#     ttl = title or f"Top-{n} minus Bottom-{n} (ann. mean={ann:.2%})"
#     plt.title(ttl)
#     plt.axhline(1.0, ls="--", color="gray")
#     plt.ylabel("Growth of $1")
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
#     plt.tight_layout()
#     _savefig_pdf(f"tb_spread_n{n}", ttl)
#     plt.show()


# def plot_vol_scaling_series(rets: pd.DataFrame, weights: pd.DataFrame, lookback=12, vol_target=0.12, smooth=4):
#     """
#     Recomputes and plots the same scaling factors implied by your scale_to_target_vol logic.
#     Shows only 2018-2024 in the plot.
#     """
#     scalers = []
#     for t in weights.index:
#         if t not in rets.index:
#             scalers.append(1.0)
#             continue
#         window = rets.loc[:t].tail(lookback)
#         port = window.dot(weights.loc[t])
#         sigma = port.std() * np.sqrt(12)
#         s = (vol_target / sigma) if sigma and sigma > 0 else 1.0
#         s = float(np.clip(s, 0.25, 4.0))
#         scalers.append(s)

#     s = pd.Series(scalers, index=weights.index)
#     s_smooth = s.rolling(smooth, min_periods=1, center=True).mean()
#     start_date = "2018-01-01"
#     end_date = "2024-12-31"
#     s_plot = s.loc[start_date:end_date]
#     s_smooth_plot = s_smooth.loc[start_date:end_date]
#     ax = s_plot.plot(alpha=0.3, label="Scaling")
#     s_smooth_plot.plot(ax=ax, label=f"Smoothed ({smooth})", linewidth=2)
#     plt.title("Volatility Target Scaling Factor")
#     plt.axhline(1.0, ls="--", color="gray")
#     plt.ylabel("Scaling Factor")
#     plt.xlabel("Date")
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     _savefig_pdf("vol_scaling_series", "vol_targeting")
#     plt.show()

# def plot_sector_weights(weights: pd.DataFrame, top_k: int | None = None):
#     """
#     Stacked area of sector weights over time with signed positions:
#       - Top panel: stacked LONG weights (>=0)
#       - Bottom panel: stacked SHORT weights (shown below zero)
#     """
#     if weights is None or weights.empty:
#         raise ValueError("plot_sector_weights: 'weights' is empty or None.")

#     W = weights.copy().fillna(0.0)

#     # Optionally keep only the most active sectors by average |weight|
#     if top_k is not None and top_k < W.shape[1]:
#         order = W.abs().mean().sort_values(ascending=False).index[:top_k]
#         W = W[order]

#     # Filter dates from 2010 to 2024 for plotting
#     start_date = "2010-01-01"
#     end_date = "2024-12-31"
#     W_plot = W.loc[start_date:end_date]

#     # Split into positive and negative parts
#     W_pos = W_plot.clip(lower=0.0)
#     W_neg_mag = -W_plot.clip(upper=0.0)  # positive magnitudes of shorts

#     if W_pos.sum().sum() == 0 and W_neg_mag.sum().sum() == 0:
#         raise ValueError("plot_sector_weights: all weights are zero over the period.")

#     fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios':[2, 1]})

#     # LONGS (stacked above zero)
#     if W_pos.sum().sum() > 0:
#         axes[0].stackplot(
#             W_pos.index,
#             *[W_pos[c].values for c in W_pos.columns],
#             labels=W_pos.columns
#         )
#     axes[0].axhline(0.0, color="black", linewidth=0.8)
#     axes[0].set_title("Sector Weight Dynamics — LONGS (stacked)")
#     axes[0].set_ylabel("Weight")
#     axes[0].legend(ncol=min(5, len(W_pos.columns)), fontsize=8, loc="upper left")

#     # SHORTS (stacked below zero)
#     if W_neg_mag.sum().sum() > 0:
#         axes[1].stackplot(
#             W_neg_mag.index,
#             *[W_neg_mag[c].values for c in W_neg_mag.columns],
#             labels=W_neg_mag.columns
#         )
#         axes[1].invert_yaxis()  # show below zero visually
#     axes[1].axhline(0.0, color="black", linewidth=0.8)
#     axes[1].set_title("Sector Weight Dynamics — SHORTS (stacked, below zero)")
#     axes[1].set_ylabel("Magnitude")
#     axes[1].set_xlabel("Date")

#     # Set x-axis ticks yearly from 2010 to 2024
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     axes[1].set_xticks(years)
#     axes[1].set_xticklabels([year.year for year in years], rotation=45)

#     plt.tight_layout()
#     _savefig_pdf("sector_weights", "long_short_stacked")
#     plt.show()


# def _turnover_series(weights: pd.DataFrame) -> pd.Series:
#     """Turnover at t = sum |w_t - w_{t-1}|."""
#     w_shift = weights.shift(1).fillna(0)
#     to = (weights.fillna(0) - w_shift).abs().sum(axis=1)
#     return to

# def plot_turnover(weights: pd.DataFrame):
#     # Compute turnover series on full data
#     to = _turnover_series(weights)
    
#     # Filter for plotting 2010-01-01 to 2024-12-31
#     start_date = "2010-01-01"
#     end_date = "2024-12-31"
#     to_plot = to.loc[start_date:end_date]
    
#     # Time series plot
#     ax = to_plot.plot(title="Monthly Turnover", lw=1.3)
#     plt.ylabel("Turnover")
#     plt.xlabel("Date")
    
#     # Set x-axis ticks yearly from 2010 to 2024
#     years = pd.date_range(start=start_date, end=end_date, freq='YS')
#     ax.set_xticks(years)
#     ax.set_xticklabels([year.year for year in years], rotation=45)
    
#     plt.tight_layout()
#     _savefig_pdf("turnover_timeseries")
#     plt.show()

#     # Histogram (distribution)
#     plt.figure()
#     to_plot.hist(bins=20)
#     plt.title(f"Turnover Distribution (mean={to_plot.mean():.2f}, median={to_plot.median():.2f})")
#     plt.xlabel("Turnover")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     _savefig_pdf("turnover_hist")
#     plt.show()


# def plot_subperiod_sharpe(port_net: pd.Series,
#                           cuts=("2009-01","2015-01","2020-01","2024-12")):
#     """Bars of Sharpe per subperiod defined by cut dates."""
#     r = port_net.dropna()
#     dates = list(pd.to_datetime(cuts))
#     if dates[0] > r.index.min(): dates[0] = r.index.min()
#     if dates[-1] < r.index.max(): dates[-1] = r.index.max()
#     labs, vals = [], []
#     for i in range(len(dates)-1):
#         s = r.loc[dates[i]:dates[i+1]]
#         if s.empty: continue
#         sh = (s.mean()/s.std()) * np.sqrt(12) if s.std() else np.nan
#         labs.append(f"{dates[i].date()}–{dates[i+1].date()}"); vals.append(sh)
#     plt.bar(labs, vals)
#     plt.title("Sharpe by Subperiod"); plt.xticks(rotation=30, ha="right")
#     for i,v in enumerate(vals):
#         plt.text(i, v + (0.02 if v>=0 else -0.06), f"{v:.2f}", ha="center")
#     plt.tight_layout()
#     _savefig_pdf("subperiod_sharpe")
#     plt.show()


# def plot_return_distribution(port_net: pd.Series):
#     r = port_net.dropna()
#     r.plot(kind="hist", bins=30, density=True, alpha=0.5, title="Monthly Return Distribution")
#     r.plot(kind="kde")
#     plt.xlabel("Monthly Return")
#     plt.tight_layout()
#     _savefig_pdf("return_distribution")
#     plt.show()


# diagnostics.py
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

_CANON_FACTORS = ["Mkt_RF","SMB","HML","RMW","CMA","Mom","RF"]

# ========== MANUAL DATE CONTROLS ==========
# Set the actual trading period (after inception masks are applied)
PLOT_START_DATE = "2010-03-01"  # First valid trading month
PLOT_END_DATE = "2024-12-31"    # End of backtest

# Helper function for consistent x-axis ticks across all plots
def _get_year_ticks(start_date_str, end_date_str):
    """Generate year ticks starting from the year of the start date."""
    start_year = int(start_date_str[:4])
    tick_start = f"{start_year}-01-01"  # Always show the year of the start date
    return pd.date_range(start=tick_start, end=end_date_str, freq='YS')
# ==========================================


# -------------------------- SAVE HELPERS (PDF) -------------------------- #
def _safe_name(s: str) -> str:
    """Sanitize a string to be a safe filename."""
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "figure"

def _savefig_pdf(name: str, title: str | None = None):
    """
    Save current matplotlib figure to FIG_DIR as PDF (tight layout).
    File name is derived from `name` (and optional title).
    """
    fname = _safe_name(name if title is None else f"{name}_{title}")
    path = os.path.join(FIG_DIR, f"{fname}.pdf")
    plt.savefig(path, bbox_inches="tight")
    # (If you ever want PNGs too, add: plt.savefig(path.replace('.pdf','.png'), dpi=300))


# -------------------------- FACTOR STANDARDIZER -------------------------- #
def _standardize_ff_columns(ff: pd.DataFrame) -> pd.DataFrame:
    """Map various FF column variants to canonical names and fix percent/decimal scale."""
    colmap = {}
    for c in ff.columns:
        cl = c.strip().lower()
        if cl in ("mkt-rf","mkt_rf","mktrf","market-rf","market_rf","mkt_rf (%)","mkt minus rf","excess market return"):
            colmap[c] = "Mkt_RF"
        elif cl == "smb":
            colmap[c] = "SMB"
        elif cl == "hml":
            colmap[c] = "HML"
        elif cl == "rmw":
            colmap[c] = "RMW"
        elif cl == "cma":
            colmap[c] = "CMA"
        elif cl in ("mom","umd","momentum"):
            colmap[c] = "Mom"
        elif cl in ("rf","riskfree","risk_free","risk-free","risk free"):
            colmap[c] = "RF"

    ff2 = ff.rename(columns=colmap).copy()

    # Convert percent -> decimal if needed (heuristic on numeric magnitude)
    num = ff2.select_dtypes(include=[np.number])
    if not num.empty and num.abs().mean().mean() > 0.5:
        for c in num.columns:
            ff2[c] = ff2[c] / 100.0

    return ff2


# -------------------------- CORE PLOTS (PDF) -------------------------- #
def plot_equity_curves(df, title="Strategy Cumulative Returns"):
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # Filter to manual trading period
    df_filtered = df.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    # Plot cumulative returns
    ax = (1 + df_filtered.fillna(0)).cumprod().plot(title=title, lw=1.3)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    
    # Set x-axis ticks yearly (including the start year)
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    
    plt.tight_layout()
    _savefig_pdf("equity_curve", title)
    plt.show()

def plot_drawdown(r, title="Drawdown"):
    # Filter to manual trading period
    r_filtered = r.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    eq = (1 + r_filtered).cumprod()
    dd = eq / eq.cummax() - 1
    
    ax = dd.plot(title=title, color="tomato", lw=1.3)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    
    plt.tight_layout()
    _savefig_pdf("drawdown", title)
    plt.show()


def plot_rolling_sharpe(r, window=24):
    title = f"Rolling Sharpe ({window}M)"
    
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(r.index):
        r.index = pd.to_datetime(r.index)
    
    # Filter to manual trading period FIRST, then compute rolling on filtered data
    r_filtered = r.loc[PLOT_START_DATE:PLOT_END_DATE]
    roll_sharpe = (r_filtered.rolling(window).mean() / r_filtered.rolling(window).std()) * np.sqrt(12)
    
    ax = roll_sharpe.plot(title=title, lw=1.3, color="steelblue")
    plt.axhline(0, color="gray", ls="--")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Date")
    
    # Set x-axis ticks yearly (including the start year)
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    
    plt.tight_layout()
    _savefig_pdf(f"rolling_sharpe_w{window}", title)
    plt.show()


def plot_rolling_alpha(port_df: pd.DataFrame, ff: pd.DataFrame, window: int = 24, title: str = None):
    """
    Rolling alpha (OLS, no HAC) vs. FF5 + Mom.
    - port_df: must contain 'Net' or 'net'
    - ff: FF factors DataFrame with any reasonable column naming; this function normalizes them
    """
    # Normalize portfolio column name
    p = port_df.copy()
    if "Net" in p.columns:
        p = p.rename(columns={"Net": "net"})
    elif "net" not in p.columns:
        raise KeyError("Portfolio DataFrame must contain 'Net' or 'net' column.")

    # Standardize factor names and scale
    ff2 = _standardize_ff_columns(ff)

    # Ensure we have at least RF and Market
    needed_min = {"RF", "Mkt_RF"}
    if not needed_min.issubset(set(ff2.columns)):
        missing = needed_min - set(ff2.columns)
        raise KeyError(f"Missing required factor column(s): {missing} after standardization.")

    # Join and drop NA
    aligned = p.join(ff2, how="inner").dropna(subset=["net", "RF", "Mkt_RF"])
    aligned = aligned.sort_index()

    # Choose available factor set (don't error if some are missing)
    factor_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in aligned.columns]

    # Filter to manual trading period FIRST
    aligned_filtered = aligned.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    # Compute rolling alpha series on filtered data
    alphas = []
    idx = aligned_filtered.index
    for i in range(window, len(idx) + 1):
        sub = aligned_filtered.iloc[i - window:i].copy()
        y = sub["net"] - sub["RF"]
        X = sm.add_constant(sub[factor_cols])
        try:
            mdl = sm.OLS(y, X).fit()
            alphas.append(mdl.params.get("const", np.nan))
        except Exception:
            alphas.append(np.nan)

    alpha_series = pd.Series(alphas, index=idx[window - 1:])
    if title is None:
        title = f"Rolling Alpha (Window={window} months)"

    # Plot (annualized alpha)
    ann_alpha = (1 + alpha_series) ** 12 - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ann_alpha.plot(ax=ax, title=title)
    ax.axhline(0.0, ls="--", color="gray")
    ax.set_ylabel("Annualized Alpha")
    ax.set_xlabel("Date")
    
    # Set x-axis to show every year (including the start year)
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45, ha='right')
    
    plt.tight_layout()
    _savefig_pdf(f"rolling_alpha_w{window}", title)
    plt.show()

def plot_factor_exposures(model):
    """Bar chart of factor coefficients from latest regression."""
    params = model.params.drop("const", errors="ignore")
    params.plot(kind="bar", color="mediumpurple", title="Factor Exposures (FF5 + Momentum)")
    plt.axhline(0, color="gray", ls="--")
    plt.tight_layout()
    _savefig_pdf("factor_exposures", "ff5_mom")
    plt.show()


def plot_signal_correlation(z_signals):
    """Optional: visualize correlation among standardized signals."""
    corr = z_signals.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Signal Correlation Matrix")
    plt.tight_layout()
    _savefig_pdf("signal_corr_heatmap")
    plt.show()


# -------------------------- UTILITIES -------------------------- #
def first_last_trading(df):
    """Find first and last months where the portfolio is actually active."""
    active = (df != 0).any(axis=1)
    active_dates = df.index[active]
    if len(active_dates) == 0:
        print("⚠️ No trading activity detected.")
        return
    first_trade = active_dates.min()
    last_trade  = active_dates.max()
    print(f"\n🔹 First trading month: {first_trade.strftime('%Y-%m')}")
    print(f"🔹 Last trading month:  {last_trade.strftime('%Y-%m')}")
    print(f"🔹 Total active months: {len(active_dates)}")


# -------------------------- NEW DIAGNOSTICS (PDF) -------------------------- #
def _cs_ic_series(fcast: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """
    Spearman rank IC: forecast at t vs realized sector return at t+1.
    Returns a monthly time series.
    """
    idx = fcast.index.intersection(rets.index)
    out = []
    for t in idx[:-1]:
        f = fcast.loc[t].dropna()
        r_next = rets.loc[t + pd.offsets.MonthEnd(1)].reindex(f.index)
        common = f.index.intersection(r_next.dropna().index)
        if len(common) < 5:
            out.append((t, np.nan))
            continue
        ic = f.loc[common].rank().corr(r_next.loc[common].rank(), method="spearman")
        out.append((t, ic))
    return pd.Series(dict(out)).sort_index()


def plot_cs_ic(fcast: pd.DataFrame, rets: pd.DataFrame, roll=12, title=None):
    """Cross-sectional IC with rolling mean."""
    # Compute IC series on full data
    ic = _cs_ic_series(fcast, rets)
    
    # Filter to manual trading period
    ic_plot = ic.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    ax = ic_plot.plot(alpha=0.4, label="Monthly IC")
    ic_plot.rolling(roll).mean().plot(ax=ax, linewidth=2, label=f"Rolling mean ({roll}m)")
    plt.axhline(0, ls="--", color="gray")
    
    m = ic.mean()
    s = ic.std(ddof=1)
    t = m / (s / np.sqrt(ic.dropna().shape[0])) if ic.dropna().shape[0] > 2 else np.nan
    ttl = title or f"Cross-Sectional IC (mean={m:.3f}, t≈{t:.2f})"
    plt.title(ttl)
    plt.legend()
    
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    plt.tight_layout()
    _savefig_pdf("cs_ic", ttl)
    plt.show()



def _tb_spread_series(fcast: pd.DataFrame, rets: pd.DataFrame, n=3) -> pd.Series:
    """
    Long top-n, short bottom-n by forecast at t; realized at t+1.
    Returns a monthly L/S series.
    """
    idx = fcast.index.intersection(rets.index)
    out = []
    for t in idx[:-1]:
        f = fcast.loc[t].dropna().sort_values(ascending=False)
        if len(f) < 2*n:
            out.append((t, np.nan)); continue
        top = f.index[:n]; bot = f.index[-n:]
        r_next = rets.loc[t + pd.offsets.MonthEnd(1)]
        spread = r_next[top].mean() - r_next[bot].mean()
        out.append((t, spread))
    return pd.Series(dict(out)).sort_index()


def plot_top_bottom_spread(fcast: pd.DataFrame, rets: pd.DataFrame, n=3, title=None):
    s = _tb_spread_series(fcast, rets, n=n)
    
    # Filter to manual trading period
    s_plot = s.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    ann = (1 + s.dropna().mean())**12 - 1 if s.dropna().size else np.nan
    ax = (1 + s_plot.fillna(0)).cumprod().plot(lw=1.3)
    ttl = title or f"Top-{n} minus Bottom-{n} (ann. mean={ann:.2%})"
    plt.title(ttl)
    plt.axhline(1.0, ls="--", color="gray")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    plt.tight_layout()
    _savefig_pdf(f"tb_spread_n{n}", ttl)
    plt.show()


def plot_vol_scaling_series(rets: pd.DataFrame, weights: pd.DataFrame, lookback=12, vol_target=0.12, smooth=4):
    """
    Recomputes and plots the same scaling factors implied by your scale_to_target_vol logic.
    """
    # Filter to manual trading period
    weights_plot = weights.loc[PLOT_START_DATE:PLOT_END_DATE]
    rets_plot = rets.loc[:PLOT_END_DATE]
    
    scalers = []
    for t in weights_plot.index:
        if t not in rets_plot.index:
            scalers.append(1.0)
            continue
        window = rets_plot.loc[:t].tail(lookback)
        port = window.dot(weights_plot.loc[t])
        sigma = port.std() * np.sqrt(12)
        s = (vol_target / sigma) if sigma and sigma > 0 else 1.0
        s = float(np.clip(s, 0.25, 4.0))
        scalers.append(s)

    s = pd.Series(scalers, index=weights_plot.index)
    s_smooth = s.rolling(smooth, min_periods=1, center=True).mean()
    
    ax = s.plot(alpha=0.3, label="Scaling")
    s_smooth.plot(ax=ax, label=f"Smoothed ({smooth})", linewidth=2)
    plt.title("Volatility Target Scaling Factor")
    plt.axhline(1.0, ls="--", color="gray")
    plt.ylabel("Scaling Factor")
    plt.xlabel("Date")
    
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    plt.legend()
    plt.tight_layout()
    _savefig_pdf("vol_scaling_series", "vol_targeting")
    plt.show()

def plot_sector_weights(weights: pd.DataFrame, top_k: int | None = None):
    """
    Stacked area of sector weights over time with signed positions:
      - Top panel: stacked LONG weights (>=0)
      - Bottom panel: stacked SHORT weights (shown below zero)
    """
    if weights is None or weights.empty:
        raise ValueError("plot_sector_weights: 'weights' is empty or None.")

    # Filter to manual trading period
    W = weights.loc[PLOT_START_DATE:PLOT_END_DATE].copy().fillna(0.0)

    # Optionally keep only the most active sectors by average |weight|
    if top_k is not None and top_k < W.shape[1]:
        order = W.abs().mean().sort_values(ascending=False).index[:top_k]
        W = W[order]

    # Split into positive and negative parts
    W_pos = W.clip(lower=0.0)
    W_neg_mag = -W.clip(upper=0.0)  # positive magnitudes of shorts

    if W_pos.sum().sum() == 0 and W_neg_mag.sum().sum() == 0:
        raise ValueError("plot_sector_weights: all weights are zero over the period.")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 7), gridspec_kw={'height_ratios':[2, 1]})

    # LONGS (stacked above zero)
    if W_pos.sum().sum() > 0:
        axes[0].stackplot(
            W_pos.index,
            *[W_pos[c].values for c in W_pos.columns],
            labels=W_pos.columns
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Sector Weight Dynamics — LONGS (stacked)")
    axes[0].set_ylabel("Weight")
    axes[0].legend(ncol=min(5, len(W_pos.columns)), fontsize=8, loc="upper left")

    # SHORTS (stacked below zero)
    if W_neg_mag.sum().sum() > 0:
        axes[1].stackplot(
            W_neg_mag.index,
            *[W_neg_mag[c].values for c in W_neg_mag.columns],
            labels=W_neg_mag.columns
        )
        axes[1].invert_yaxis()  # show below zero visually
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Sector Weight Dynamics — SHORTS (stacked, below zero)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlabel("Date")

    # Set x-axis ticks yearly (including the start year)
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    axes[1].set_xticks(years)
    axes[1].set_xticklabels([year.year for year in years], rotation=45)

    plt.tight_layout()
    _savefig_pdf("sector_weights", "long_short_stacked")
    plt.show()


def _turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Turnover at t = sum |w_t - w_{t-1}|."""
    w_shift = weights.shift(1).fillna(0)
    to = (weights.fillna(0) - w_shift).abs().sum(axis=1)
    return to

def plot_turnover(weights: pd.DataFrame):
    # Filter to manual trading period
    weights_plot = weights.loc[PLOT_START_DATE:PLOT_END_DATE]
    
    # Compute turnover series
    to = _turnover_series(weights_plot)
    
    # Time series plot
    ax = to.plot(title="Monthly Turnover", lw=1.3)
    plt.ylabel("Turnover")
    plt.xlabel("Date")
    
    # Set x-axis ticks yearly (including the start year)
    years = _get_year_ticks(PLOT_START_DATE, PLOT_END_DATE)
    ax.set_xticks(years)
    ax.set_xticklabels([year.year for year in years], rotation=45)
    
    plt.tight_layout()
    _savefig_pdf("turnover_timeseries")
    plt.show()

    # Histogram (distribution)
    plt.figure()
    to.hist(bins=20)
    plt.title(f"Turnover Distribution (mean={to.mean():.2f}, median={to.median():.2f})")
    plt.xlabel("Turnover")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _savefig_pdf("turnover_hist")
    plt.show()


def plot_subperiod_sharpe(port_net: pd.Series,
                          cuts=("2009-01","2015-01","2020-01","2024-12")):
    """Bars of Sharpe per subperiod defined by cut dates."""
    r = port_net.dropna()
    dates = list(pd.to_datetime(cuts))
    if dates[0] > r.index.min(): dates[0] = r.index.min()
    if dates[-1] < r.index.max(): dates[-1] = r.index.max()
    labs, vals = [], []
    for i in range(len(dates)-1):
        s = r.loc[dates[i]:dates[i+1]]
        if s.empty: continue
        sh = (s.mean()/s.std()) * np.sqrt(12) if s.std() else np.nan
        labs.append(f"{dates[i].date()}–{dates[i+1].date()}"); vals.append(sh)
    plt.bar(labs, vals)
    plt.title("Sharpe by Subperiod"); plt.xticks(rotation=30, ha="right")
    for i,v in enumerate(vals):
        plt.text(i, v + (0.02 if v>=0 else -0.06), f"{v:.2f}", ha="center")
    plt.tight_layout()
    _savefig_pdf("subperiod_sharpe")
    plt.show()


def plot_return_distribution(port_net: pd.Series):
    r = port_net.dropna()
    if r.empty:
        print("No valid return data to plot")
        return
    
    r.plot(kind="hist", bins=30, density=True, alpha=0.5, title="Monthly Return Distribution")
    r.plot(kind="kde")
    plt.xlabel("Monthly Return")
    plt.tight_layout()
    _savefig_pdf("return_distribution")
    plt.show()