import numpy as np
import  pandas as pd

def ann_return(r):
    n =len(r.dropna())
    if n==0:
        return np.nan
    return (1 + r).prod() ** (12 / n) - 1

def ann_vol(r):
    return r.std() * np.sqrt(12)

def sharpe(r, rf = 0.0):
    ex = r-rf
    vol = ann_vol(ex)
    if vol ==0 or np.isnan(vol):
        return np.nan
    return ex.mean()*12/ vol

def max_drawdown(r):
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq - peak)/ peak
    return dd.min()

def summary_table(ret_gross_net, rf_series=None):
    df = ret_gross_net.copy()
    stats = {}
    for col in df.columns:
        r = df[col].dropna()
        stats[col] = {
            "Ann.Return": ann_return(r),
            "Ann.Vol": ann_vol(r),
            "Sharpe(0%)": sharpe(r, 0.0),
            "MaxDD": max_drawdown(r),
        }
    if rf_series is not None and 'Net' in df.columns:
        aligned = pd.concat([df["Net"], rf_series.squeeze()], axis=1).dropna()
        stats["Net"]["Sharpe(ex RF)"] = sharpe(aligned.iloc[:,0], aligned.iloc[:,1])
    return pd.DataFrame(stats)

def extended_metrics(port_ret, ff_model=None, benchmark=None):
    """
    port_ret: monthly return series (net)
    ff_model: fitted model from benchmark.align_factors()
    benchmark: optional benchmark return series (e.g., S&P500)
    """
    import numpy as np
    import pandas as pd

    r = port_ret.dropna()
    mean_r = r.mean()
    std_r = r.std()
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = std_r * np.sqrt(12)
    sharpe = mean_r / std_r * np.sqrt(12)
    sortino = (mean_r * 12) / (r[r < 0].std() * np.sqrt(12))
    hit_ratio = (r > 0).mean()
    skew = r.skew()
    kurt = r.kurtosis()

    res = {
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Hit Ratio": hit_ratio,
        "Skew": skew,
        "Kurtosis": kurt
    }

    if benchmark is not None:
        diff = (r - benchmark.reindex_like(r)).dropna()
        tracking_err = diff.std() * np.sqrt(12)
        ir = (ann_ret - ((1 + benchmark).prod() ** (12 / len(benchmark)) - 1)) / tracking_err
        res.update({"Tracking Error": tracking_err, "Information Ratio": ir})

    if ff_model is not None:
        a = ff_model.params.get("const", np.nan)
        beta_mkt = ff_model.params.get("Mkt_RF", np.nan)
        res.update({
            "Alpha (annual)": (1 + a)**12 - 1 if not np.isnan(a) else np.nan,
            "Market Beta": beta_mkt
        })
    return pd.Series(res)
