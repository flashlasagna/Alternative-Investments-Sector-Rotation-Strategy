import pandas as pd
from config import TCOST_BPS

def turnover(prev_w, new_w):

    prev_w = prev_w.reindex(new_w.index).fillna(0.0)
    return (prev_w - new_w).abs().sum()

def apply_tcosts(port_ret_series, weights_df, tcost_bps=0):
    if tcost_bps==0:
        return port_ret_series

    costs = []
    prev = None
    for t, w in weights_df.iterrows():
        if prev is None:
            costs.append(0.0)
        else:
            to = turnover((tcost_bps/10000)*to)
        prev = w
    costs = pd.Series(costs, index=weights_df.index)

    return port_ret_series-costs
def simulate_portfolio(monthly_rets, weight_df):
    """
        Weights decided at t (based on signals up to t) are applied for returns at t+1.
    """
    idx = monthly_rets.index.intersection(weight_df.index)
    W = weight_df.loc[idx]
    R = monthly_rets.loc[idx]

    W_impl = W.shift(1).fillna(0.0)
    port_ret = (W_impl*R).sum(axis=1)
    port_ret_net = apply_tcosts(port_ret, W_impl, tcost_bps=TCOST_BPS)
    return pd.DataFrame({"Gross": port_ret, "Net": port_ret_net})