# diagnostics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)
_CANON_FACTORS = ["Mkt_RF","SMB","HML","RMW","CMA","Mom","RF"]

def _standardize_ff_columns(ff: pd.DataFrame) -> pd.DataFrame:
    """Map various FF column variants to canonical names and fix percent/decimal scale."""
    colmap = {}
    for c in ff.columns:
        cl = c.strip().lower()
        if cl in ("mkt-rf","mkt_rf","mktrf","market-rf","market_rf","mkt_rf (%)"):
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
        elif cl in ("rf","riskfree","risk_free","risk-free"):
            colmap[c] = "RF"

    ff2 = ff.rename(columns=colmap).copy()

    # Convert percent -> decimal if needed (heuristic on numeric magnitude)
    num = ff2.select_dtypes(include=[np.number])
    if not num.empty and num.abs().mean().mean() > 0.5:
        for c in num.columns:
            ff2[c] = ff2[c] / 100.0

    return ff2

def plot_equity_curves(df, title="Strategy Cumulative Returns"):
    (1 + df.fillna(0)).cumprod().plot(title=title, lw=1.3)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/equity_curve.png", dpi=300)
    plt.show()


def plot_drawdown(r, title="Drawdown"):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    dd.plot(title=title, color="tomato", lw=1.3)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/drawdown.png", dpi=300)
    plt.show()


def plot_rolling_sharpe(r, window=24):
    roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(12)
    roll_sharpe.plot(title=f"Rolling Sharpe ({window}M)", lw=1.3, color="steelblue")
    plt.axhline(0, color="gray", ls="--")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rolling_sharpe.png", dpi=300)
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

    # Choose available factor set (don’t error if some are missing)
    factor_cols = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","Mom"] if c in aligned.columns]

    # Compute rolling alpha series
    alphas = []
    idx = aligned.index
    for i in range(window, len(idx) + 1):
        sub = aligned.iloc[i - window:i].copy()
        y = sub["net"] - sub["RF"]
        X = sm.add_constant(sub[factor_cols])
        try:
            mdl = sm.OLS(y, X).fit()
            alphas.append(mdl.params.get("const", np.nan))
        except Exception:
            alphas.append(np.nan)

    alpha_series = pd.Series(alphas, index=idx[window - 1:])
    if title is None:
        title = f"Rolling Alpha (window={window} months)"

    # Plot (annualized alpha)
    ann_alpha = (1 + alpha_series) ** 12 - 1
    ann_alpha.plot(title=title)
    plt.axhline(0.0, ls="--", color="gray")
    plt.ylabel("Annualized Alpha")
    plt.tight_layout()
    plt.show()


def plot_factor_exposures(model):
    """Bar chart of factor coefficients from latest regression."""
    params = model.params.drop("const", errors="ignore")
    params.plot(kind="bar", color="mediumpurple", title="Factor Exposures (FF5 + Momentum)")
    plt.axhline(0, color="gray", ls="--")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/factor_exposures.png", dpi=300)
    plt.show()


def plot_signal_correlation(z_signals):
    """Optional: visualize correlation among standardized signals."""
    corr = z_signals.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Signal Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/signal_corr_heatmap.png", dpi=300)
    plt.show()

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
    last_trade  = active_dates.max()
    print(f"\n🔹 First trading month: {first_trade.strftime('%Y-%m')}")
    print(f"🔹 Last trading month:  {last_trade.strftime('%Y-%m')}")

    # Optional: show how many months traded
    print(f"🔹 Total active months: {len(active_dates)}")

