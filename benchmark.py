import pandas as pd
import statsmodels.api as sm

def align_factors(port_ret_df, ff_df):
    """
       port_ret_df: DataFrame with 'Net' monthly total return.
       ff_df: monthly factors with columns including ['Mkt_RF','SMB','HML','RMW','CMA','RF','Mom'] possibly in percent.
    """
    # Convert percent -> decimal if needed
    sample = ff_df.select_dtypes(include="number")
    if sample.abs().mean().mean() > 0.5:
        ff_df = ff_df / 100

    # Build X with available columns
    avail = ff_df.columns
    Xcols = [c for c in ['Mkt_RF','SMB','HML','RMW','CMA','Mom'] if c in avail]

    aligned = port_ret_df.join(ff_df, how='inner')
    y = aligned['Net'] - aligned['RF']
    X = aligned[Xcols]
    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags":6})
    return model

def describe_alpha(model):
    a = model.params.get('const',None)
    return {
        "alpha annualized": None if a is None else (1 + a) ** 12 - 1,
        "alpha tstat": model.tvalues.get('const',None),
        "r2": model.rsquared_adj
    }
