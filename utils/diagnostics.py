# diagnostics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)


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


def plot_rolling_alpha(r, ff_df, window=24):
    """Compute rolling alpha (annualized) using FF5+Mom factors."""
    aligned = r.join(ff_df, how="inner").dropna()
    alphas = []
    for i in range(window, len(aligned)):
        sub = aligned.iloc[i - window:i]
        y = sub["net"] - sub["RF"]
        X = sub[["Mkt_RF","SMB","HML","RMW","CMA","Mom"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        alpha_ann = (1 + model.params["const"])**12 - 1
        alphas.append(alpha_ann)
    idx = aligned.index[window:]
    pd.Series(alphas, index=idx).plot(title=f"Rolling Alpha ({window}M)", color="darkgreen", lw=1.3)
    plt.axhline(0, color="gray", ls="--")
    plt.ylabel("Alpha (annualized)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rolling_alpha.png", dpi=300)
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
