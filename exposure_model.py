# exposure_model.py
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import Ridge
import config


def _align_panels(monthly_rets: pd.DataFrame,
                  z_signals: pd.DataFrame,
                  assume_signals_already_lagged: bool = True):
    """
    Align returns and signal panels on a common monthly, month-end index.
    If assume_signals_already_lagged=False, we shift signals by 1M here.
    """
    idx = monthly_rets.index.intersection(z_signals.index)
    r = monthly_rets.loc[idx]
    X = z_signals.loc[idx]
    if not assume_signals_already_lagged:
        X = X.shift(1)
    return r, X


def rolling_sectors_betas_ols(monthly_rets: pd.DataFrame,
                              z_signals: pd.DataFrame,
                              lookback_months: int = 36,
                              min_months: int = 24,
                              assume_signals_already_lagged: bool = True):
    """
    Rolling OLS per sector:
        r_{i,t} = α_i,t + β_i,t' z_t + ε_t
    IMPORTANT: z_signals are expected to be availability-lagged upstream.
    """
    r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)

    betas = {}
    for sector in r.columns:
        y = r[sector]
        df = pd.concat([y, X], axis=1).dropna(how="any")
        if df.empty:
            betas[sector] = pd.DataFrame()
            continue

        rows, idx = [], df.index
        for i in range(lookback_months - 1, len(idx)):
            window = idx[i - (lookback_months - 1): i + 1]
            sub = df.loc[window]
            if sub.shape[0] < min_months:
                continue

            y_win = sub.iloc[:, 0]
            X_win = add_constant(sub.iloc[:, 1:])  # adds 'const'
            try:
                mdl = OLS(y_win.values, X_win.values).fit()
                params = pd.Series(mdl.params, index=X_win.columns, name=window[-1])
                rows.append(params)
            except Exception:
                pass

        betas[sector] = pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()

    return betas


def rolling_sectors_betas_ridge(monthly_rets: pd.DataFrame,
                                z_signals: pd.DataFrame,
                                lookback_months: int = 36,
                                min_months: int = 24,
                                alpha: float | None = None,
                                assume_signals_already_lagged: bool = True):
    """
    Rolling Ridge per sector (L2 regularization):
        r_{i,t} = α_i,t + β_i,t' z_t + ε_t
    Intercept is fit (not penalized). z_signals are expected to be lagged already.
    """
    if alpha is None:
        alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))

    r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)

    betas = {}
    for sector in r.columns:
        y = r[sector]
        df = pd.concat([y, X], axis=1).dropna(how="any")
        if df.empty:
            betas[sector] = pd.DataFrame()
            continue

        rows, idx = [], df.index
        for i in range(lookback_months - 1, len(idx)):
            window = idx[i - (lookback_months - 1): i + 1]
            sub = df.loc[window]
            if sub.shape[0] < min_months:
                continue

            y_win = sub.iloc[:, 0].values
            X_win = sub.iloc[:, 1:]

            mdl = Ridge(alpha=alpha, fit_intercept=True)  # intercept NOT penalized
            mdl.fit(X_win.values, y_win)

            coef_names = list(X_win.columns)
            params = pd.Series(
                {'const': float(mdl.intercept_), **dict(zip(coef_names, mdl.coef_))},
                name=window[-1]
            )
            rows.append(params)

        betas[sector] = pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()

    return betas


def rolling_sectors_betas(monthly_rets: pd.DataFrame,
                          z_signals: pd.DataFrame,
                          lookback_months: int = 36,
                          min_months: int = 24,
                          assume_signals_already_lagged: bool = True):
    """
    Dispatcher: choose OLS or Ridge based on config.USE_RIDGE.
    """
    use_ridge = bool(getattr(config, "USE_RIDGE", False))
    if use_ridge:
        alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))
        print(f"[INFO] Rolling Ridge betas (alpha={alpha})")
        return rolling_sectors_betas_ridge(
            monthly_rets, z_signals,
            lookback_months=lookback_months,
            min_months=min_months,
            alpha=alpha,
            assume_signals_already_lagged=assume_signals_already_lagged
        )
    else:
        print("[INFO] Rolling OLS betas")
        return rolling_sectors_betas_ols(
            monthly_rets, z_signals,
            lookback_months=lookback_months,
            min_months=min_months,
            assume_signals_already_lagged=assume_signals_already_lagged
        )


def sector_expected_returns(betas_dict: dict[str, pd.DataFrame],
                            z_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute \hat r_{i,t+1} = const_{i,t} + β_{i,t}' z_t, for each sector i and month t.
    z_signals are the info known at t (already lagged upstream).
    """
    idx = z_signals.index
    sectors = list(betas_dict.keys())
    out = pd.DataFrame(index=idx, columns=sectors, dtype=float)

    for sector, bdf in betas_dict.items():
        if bdf is None or bdf.empty:
            continue
        cols = [c for c in bdf.columns if c != "const" and c in z_signals.columns]
        t_common = bdf.index.intersection(idx)
        if not cols or t_common.empty:
            continue

        # row-wise dot product + intercept (vectorized)
        dot = (bdf.loc[t_common, cols] * z_signals.loc[t_common, cols]).sum(axis=1)
        const = bdf.loc[t_common, "const"] if "const" in bdf.columns else 0.0
        out.loc[t_common, sector] = dot + const

    return out