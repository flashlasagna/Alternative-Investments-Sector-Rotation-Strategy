import matplotlib.pyplot as plt
import numpy as np

def equity_curve(r, title="Equity Curve"):
    eq = (1 + r.fillna(0)).cumprod()
    eq.plot(title=title)
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.show()

def plot_equity_curves(df, title="Equity Curves"):
    (1 + df.fillna(0)).cumprod().plot(title=title)
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(r, title="Drawdown"):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    dd.plot(title=title)
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()

def plot_rolling_sharpe(r, window=24, title="Rolling Sharpe (24M)"):
    roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(12)
    roll_sharpe.plot(title=title)
    plt.axhline(0, color="gray", ls="--")
    plt.tight_layout()
    plt.show()
