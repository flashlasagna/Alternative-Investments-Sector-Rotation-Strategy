# # exposure_model.py
# import numpy as np
# import pandas as pd
# from statsmodels.api import OLS, add_constant
# from sklearn.linear_model import Ridge
# import config


# def _align_panels(monthly_rets: pd.DataFrame,
#                   z_signals: pd.DataFrame,
#                   assume_signals_already_lagged: bool = True):
#     """
#     Align returns and signal panels on a common monthly, month-end index.
#     If assume_signals_already_lagged=False, we shift signals by 1M here.
#     """
#     idx = monthly_rets.index.intersection(z_signals.index)
#     r = monthly_rets.loc[idx]
#     X = z_signals.loc[idx]
#     if not assume_signals_already_lagged:
#         X = X.shift(1)
#     return r, X

# # if we are setting USE_RIDGE = False in config.py, we are ussing this OLS function
# def rolling_sectors_betas_ols(monthly_rets: pd.DataFrame,
#                               z_signals: pd.DataFrame,
#                               lookback_months: int = 36,
#                               min_months: int = 24,
#                               assume_signals_already_lagged: bool = True):
#     """
#     Rolling OLS per sector:
#         r_{i,t} = α_i,t + β_i,t' z_t + ε_t
#     IMPORTANT: z_signals are expected to be availability-lagged upstream.
#     """
#     r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)

#     betas = {}
#     for sector in r.columns:
#         y = r[sector]
#         df = pd.concat([y, X], axis=1).dropna(how="any")
#         if df.empty:
#             betas[sector] = pd.DataFrame()
#             continue

#         rows, idx = [], df.index
#         for i in range(lookback_months - 1, len(idx)):
#             window = idx[i - (lookback_months - 1): i + 1]
#             sub = df.loc[window]
#             if sub.shape[0] < min_months:
#                 continue

#             y_win = sub.iloc[:, 0]
#             X_win = add_constant(sub.iloc[:, 1:])  # adds 'const'
#             try:
#                 mdl = OLS(y_win.values, X_win.values).fit()
#                 params = pd.Series(mdl.params, index=X_win.columns, name=window[-1])
#                 rows.append(params)
#             except Exception:
#                 pass

#         betas[sector] = pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()

#     return betas

# # if we are setting USE_RIDGE = True in config.py, we are ussing this Ridge function

# def rolling_sectors_betas_ridge(monthly_rets: pd.DataFrame,
#                                 z_signals: pd.DataFrame,
#                                 lookback_months: int = 36,
#                                 min_months: int = 24,
#                                 alpha: float | None = None,
#                                 assume_signals_already_lagged: bool = True):
#     """
#     Rolling Ridge per sector (L2 regularization):
#         r_{i,t} = α_i,t + β_i,t' z_t + ε_t
#     Intercept is fit (not penalized). z_signals are expected to be lagged already.
#     """
#     if alpha is None:
#         alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))

#     r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)

#     betas = {}
#     for sector in r.columns:
#         y = r[sector]
#         df = pd.concat([y, X], axis=1).dropna(how="any")
#         if df.empty:
#             betas[sector] = pd.DataFrame()
#             continue

#         rows, idx = [], df.index
#         for i in range(lookback_months - 1, len(idx)):
#             window = idx[i - (lookback_months - 1): i + 1]
#             sub = df.loc[window]
#             if sub.shape[0] < min_months:
#                 continue

#             y_win = sub.iloc[:, 0].values
#             X_win = sub.iloc[:, 1:]

#             mdl = Ridge(alpha=alpha, fit_intercept=True)  # intercept NOT penalized
#             mdl.fit(X_win.values, y_win)

#             coef_names = list(X_win.columns)
#             params = pd.Series(
#                 {'const': float(mdl.intercept_), **dict(zip(coef_names, mdl.coef_))},
#                 name=window[-1]
#             )
#             rows.append(params)

#         betas[sector] = pd.DataFrame(rows).sort_index() if rows else pd.DataFrame()

#     return betas


# def rolling_sectors_betas(monthly_rets: pd.DataFrame,
#                           z_signals: pd.DataFrame,
#                           lookback_months: int = 36,
#                           min_months: int = 24,
#                           assume_signals_already_lagged: bool = True):
#     """
#     Dispatcher: choose OLS or Ridge based on config.USE_RIDGE.
#     """
#     use_ridge = bool(getattr(config, "USE_RIDGE", False))
#     if use_ridge:
#         alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))
#         print(f"[INFO] Rolling Ridge betas (alpha={alpha})")
#         return rolling_sectors_betas_ridge(
#             monthly_rets, z_signals,
#             lookback_months=lookback_months,
#             min_months=min_months,
#             alpha=alpha,
#             assume_signals_already_lagged=assume_signals_already_lagged
#         )
#     else:
#         print("[INFO] Rolling OLS betas")
#         return rolling_sectors_betas_ols(
#             monthly_rets, z_signals,
#             lookback_months=lookback_months,
#             min_months=min_months,
#             assume_signals_already_lagged=assume_signals_already_lagged
#         )


# def sector_expected_returns(betas_dict: dict[str, pd.DataFrame],
#                             z_signals: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute \hat r_{i,t+1} = const_{i,t} + β_{i,t}' z_t, for each sector i and month t.
#     z_signals are the info known at t (already lagged upstream).
#     """
#     idx = z_signals.index
#     sectors = list(betas_dict.keys())
#     out = pd.DataFrame(index=idx, columns=sectors, dtype=float)

#     for sector, bdf in betas_dict.items():
#         if bdf is None or bdf.empty:
#             continue
#         cols = [c for c in bdf.columns if c != "const" and c in z_signals.columns]
#         t_common = bdf.index.intersection(idx)
#         if not cols or t_common.empty:
#             continue

#         # row-wise dot product + intercept (vectorized)
#         dot = (bdf.loc[t_common, cols] * z_signals.loc[t_common, cols]).sum(axis=1)
#         const = bdf.loc[t_common, "const"] if "const" in bdf.columns else 0.0
#         out.loc[t_common, sector] = dot + const

#     return out


# exposure_model.py
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
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


def _drop_zero_variance(X: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """
    Drop columns with zero (or near-zero) variance to avoid NaNs in correlation/PCA.
    
    Parameters:
    - X: feature matrix
    - tol: minimum standard deviation threshold
    
    Returns:
    - DataFrame with only non-constant columns
    """
    if X.empty:
        return X
    std_vals = X.std()
    keep_cols = std_vals[std_vals > tol].index.tolist()
    return X[keep_cols]


def _preprune_duplicates(X: pd.DataFrame, thresh: float = 0.98) -> pd.DataFrame:
    """
    Remove near-duplicate features within the current window.
    If two features have |correlation| > thresh, drop one.
    
    NOTE: This uses a "keep-first" strategy: when multiple features are highly
    correlated, we keep the first one in column order and drop the rest.
    This is acceptable since thresh=0.98 is very tight (literal duplicates only).
    
    Parameters:
    - X: feature matrix (should have non-zero variance columns only)
    - thresh: correlation threshold for considering features as duplicates
    
    Returns:
    - DataFrame with near-duplicate features removed
    """
    if X.shape[1] < 2:
        return X
    
    corr = X.corr().abs()
    # Set diagonal and lower triangle to 0 to avoid self-correlation and double-counting
    for i in range(len(corr)):
        for j in range(i + 1):
            corr.iloc[i, j] = 0
    
    # Find pairs with correlation > thresh
    to_drop = set()
    for col in corr.columns:
        if col in to_drop:
            continue
        # Get features highly correlated with this one
        high_corr = corr.index[corr[col] > thresh].tolist()
        # Drop all but keep the current one (keep-first strategy)
        to_drop.update(high_corr)
    
    keep_cols = [c for c in X.columns if c not in to_drop]
    return X[keep_cols]


def _fit_pca_for_window(X: pd.DataFrame,
                         var_target: float = 0.80,
                         k_max: int = 6) -> tuple:
    """
    Fit PCA on features X for the current window and return reusable components.
    This should be called ONCE per window, then used for all sectors.
    
    Signals are dropped where they have NaN; rows used for PCA are returned
    to ensure sector returns are aligned to the same row subset.
    
    Parameters:
    - X: feature matrix (n_samples, n_features) as DataFrame
    - var_target: target cumulative explained variance
    - k_max: maximum number of PCs
    
    Returns:
    - tuple: (PC_scores_k, V_k, X_mean, k, pca_rows) or None if PCA cannot be fit
        - PC_scores_k: transformed features (n_samples, k)
        - V_k: principal component loadings (k, n_features)
        - X_mean: column means (n_features,)
        - k: number of components selected
        - pca_rows: pd.Index of rows used for PCA (after dropna)
    """
    if X.shape[1] == 0:
        return None
    
    # Drop rows with any signal NaNs (PCA requirement)
    X_complete = X.dropna(how='any')
    if X_complete.shape[0] < 2:
        return None
    
    # Center X by column means (per window)
    X_mean = X_complete.mean()
    X_centered = X_complete - X_mean
    
    # Determine K: smallest K that reaches var_target, capped by k_max, window length, and n_features
    n_samples, n_features = X_complete.shape
    k_limit = min(k_max, n_samples - 1, n_features)
    
    if k_limit < 1:
        return None
    
    # Fit PCA
    pca = PCA(n_components=k_limit)
    PC_scores = pca.fit_transform(X_centered.values)  # (n_samples, k_limit)
    
    # Choose K by cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, var_target) + 1)
    k = min(k, k_limit)
    
    if k < 1:
        k = 1
    
    # Extract first k components
    PC_scores_k = PC_scores[:, :k]
    V_k = pca.components_[:k, :]  # (k, n_features)
    
    # Return row indices used for PCA (for alignment with sector returns)
    return PC_scores_k, V_k, X_mean, k, X_complete.index


def _backproject_sector_coefficients(y: np.ndarray,
                                      PC_scores_k: np.ndarray,
                                      V_k: np.ndarray,
                                      X_mean: pd.Series,
                                      feature_names: list,
                                      use_ridge: bool = False,
                                      alpha: float = 5.0) -> pd.Series:
    """
    Fit regression on PC scores for one sector, then back-project to original features.
    
    Parameters:
    - y: target variable for this sector (n_samples,)
    - PC_scores_k: PC scores from _fit_pca_for_window (n_samples, k)
    - V_k: principal component loadings (k, n_features)
    - X_mean: feature column means (n_features,)
    - feature_names: list of original feature names
    - use_ridge: if True, fit Ridge; else OLS
    - alpha: Ridge regularization strength
    
    Returns:
    - pd.Series with 'const' and one coefficient per original feature
    """
    # Fit regression on PC scores
    if use_ridge:
        mdl = Ridge(alpha=alpha, fit_intercept=True)
        mdl.fit(PC_scores_k, y)
        intercept_pc = float(mdl.intercept_)
        coef_pc = mdl.coef_  # (k,)
    else:
        PC_with_const = add_constant(PC_scores_k)
        mdl = OLS(y, PC_with_const).fit()
        intercept_pc = float(mdl.params[0])
        coef_pc = mdl.params[1:]  # (k,)
    
    # Back-project: β_original = V_k @ β_pc
    beta_original = V_k.T @ coef_pc  # (n_features,)
    
    # Adjust intercept to account for centering:
    # intercept_final = intercept_pc - beta_original' @ X_mean
    intercept_final = intercept_pc - np.dot(beta_original, X_mean.values)
    
    # Return as Series with feature names
    result = pd.Series({'const': float(intercept_final)})
    for i, col in enumerate(feature_names):
        result[col] = float(beta_original[i])
    
    return result


def rolling_sectors_betas_ols(monthly_rets: pd.DataFrame,
                              z_signals: pd.DataFrame,
                              lookback_months: int = 36,
                              min_months: int = 24,
                              assume_signals_already_lagged: bool = True):
    """
    Rolling OLS per sector:
        r_{i,t} = α_i,t + β_i,t' z_t + ε_t
    IMPORTANT: z_signals are expected to be availability-lagged upstream.
    
    When USE_PCA=True:
      - PCA is fitted ONCE per window on complete signal rows
      - Each sector's returns are aligned to those PCA rows
      - Fit regression and back-project coefficients
    """
    r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)
    
    # Get PCA settings from config
    use_pca = bool(getattr(config, "USE_PCA", False))
    pca_var_target = float(getattr(config, "PCA_VAR_TARGET", 0.80))
    pca_k_max = int(getattr(config, "PCA_K_MAX", 6))
    pca_preprune_thresh = float(getattr(config, "PCA_PREPRUNE_THRESH", 0.98))

    idx_all = r.index
    sectors = list(r.columns)
    
    # Initialize results dict
    betas = {sector: [] for sector in sectors}
    
    # Loop over windows (compute PCA once per window, not per sector)
    for i in range(lookback_months - 1, len(idx_all)):
        window = idx_all[i - (lookback_months - 1): i + 1]
        window_date = window[-1]
        
        # Extract signals for this window
        X_win_full = X.loc[window]
        
        # PCA path: compute once for this window
        if use_pca and X_win_full.shape[1] >= 2:
            try:
                # Drop zero-variance columns first
                X_nonzero = _drop_zero_variance(X_win_full)
                if X_nonzero.shape[1] < 1:
                    # No valid features, use intercept-only for all sectors
                    for sector in sectors:
                        r_win = r[sector].loc[window]
                        params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                        betas[sector].append(params)
                    continue
                
                # Optional: pre-prune near-duplicates
                X_pruned = _preprune_duplicates(X_nonzero, thresh=pca_preprune_thresh)
                
                if X_pruned.shape[1] >= 1:
                    # Fit PCA ONCE for this window (on complete signal rows)
                    pca_result = _fit_pca_for_window(X_pruned, var_target=pca_var_target, k_max=pca_k_max)
                    
                    if pca_result is not None:
                        PC_scores_k, V_k, X_mean, k, pca_rows = pca_result
                        
                        # Now fit each sector using the PCA rows
                        for sector in sectors:
                            # Align sector returns to PCA rows and drop NaNs
                            r_full = r[sector].loc[window]
                            r_aligned = r_full.loc[pca_rows]
                            y_aligned = r_aligned.dropna()
                            
                            # Check if we have enough non-NaN data for OLS on k PCs (need k+1 rows)
                            if y_aligned.shape[0] < (k + 1):
                                # Not enough data, use intercept-only
                                params = pd.Series({'const': float(y_aligned.mean())}, name=window_date)
                                betas[sector].append(params)
                                continue
                            
                            # Slice PC scores to match non-NaN y indices
                            row_pos = pca_rows.get_indexer(y_aligned.index)
                            Z_sub = PC_scores_k[row_pos, :]
                            
                            # Regress aligned non-NaN returns on sliced PC scores
                            params = _backproject_sector_coefficients(
                                y_aligned.values, Z_sub, V_k, X_mean,
                                feature_names=list(X_pruned.columns),
                                use_ridge=False
                            )
                            params.name = window_date
                            betas[sector].append(params)
                    else:
                        # PCA failed, use intercept-only
                        for sector in sectors:
                            r_win = r[sector].loc[window]
                            params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                            betas[sector].append(params)
                else:
                    # All features pruned, use intercept-only
                    for sector in sectors:
                        r_win = r[sector].loc[window]
                        params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                        betas[sector].append(params)
            except Exception:
                # If PCA fails, skip this window for all sectors
                pass
        else:
            # Standard OLS path (no PCA): fit per sector with complete rows
            for sector in sectors:
                try:
                    # Get sector's returns for this window
                    r_win = r[sector].loc[window]
                    X_win = X_win_full
                    
                    # Align returns and signals to common non-NaN rows
                    combined = pd.concat([r_win, X_win], axis=1).dropna(how='any')
                    if combined.shape[0] < min_months:
                        continue
                    
                    y_vec = combined.iloc[:, 0].values
                    X_mat = combined.iloc[:, 1:].values
                    
                    # Fit OLS
                    X_with_const = add_constant(X_mat)
                    mdl = OLS(y_vec, X_with_const).fit()
                    
                    # Extract coefficients
                    col_names = ['const'] + list(X_win.columns)
                    params = pd.Series(dict(zip(col_names, mdl.params)), name=window_date)
                    betas[sector].append(params)
                except Exception:
                    pass
    
    # Convert lists to DataFrames
    for sector in sectors:
        betas[sector] = pd.DataFrame(betas[sector]).sort_index() if betas[sector] else pd.DataFrame()
    
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
    
    When USE_PCA=True:
      - PCA is fitted ONCE per window on complete signal rows
      - Each sector's returns are aligned to those PCA rows
      - Fit regression and back-project coefficients
    """
    if alpha is None:
        alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))

    r, X = _align_panels(monthly_rets, z_signals, assume_signals_already_lagged)
    
    # Get PCA settings from config
    use_pca = bool(getattr(config, "USE_PCA", False))
    pca_var_target = float(getattr(config, "PCA_VAR_TARGET", 0.80))
    pca_k_max = int(getattr(config, "PCA_K_MAX", 6))
    pca_preprune_thresh = float(getattr(config, "PCA_PREPRUNE_THRESH", 0.98))

    idx_all = r.index
    sectors = list(r.columns)
    
    # Initialize results dict
    betas = {sector: [] for sector in sectors}
    
    # Loop over windows (compute PCA once per window, not per sector)
    for i in range(lookback_months - 1, len(idx_all)):
        window = idx_all[i - (lookback_months - 1): i + 1]
        window_date = window[-1]
        
        # Extract signals for this window
        X_win_full = X.loc[window]
        
        # PCA path: compute once for this window
        if use_pca and X_win_full.shape[1] >= 2:
            try:
                # Drop zero-variance columns first
                X_nonzero = _drop_zero_variance(X_win_full)
                if X_nonzero.shape[1] < 1:
                    # No valid features, use intercept-only for all sectors
                    for sector in sectors:
                        r_win = r[sector].loc[window]
                        params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                        betas[sector].append(params)
                    continue
                
                # Optional: pre-prune near-duplicates
                X_pruned = _preprune_duplicates(X_nonzero, thresh=pca_preprune_thresh)
                
                if X_pruned.shape[1] >= 1:
                    # Fit PCA ONCE for this window (on complete signal rows)
                    pca_result = _fit_pca_for_window(X_pruned, var_target=pca_var_target, k_max=pca_k_max)
                    
                    if pca_result is not None:
                        PC_scores_k, V_k, X_mean, k, pca_rows = pca_result
                        
                        # Now fit each sector using the PCA rows
                        for sector in sectors:
                            # Align sector returns to PCA rows and drop NaNs
                            r_full = r[sector].loc[window]
                            r_aligned = r_full.loc[pca_rows]
                            y_aligned = r_aligned.dropna()
                            
                            # Check if we have enough non-NaN data for Ridge on k PCs (need ≥ max(2,k) rows)
                            if y_aligned.shape[0] < max(2, k):
                                # Not enough data, use intercept-only
                                params = pd.Series({'const': float(y_aligned.mean())}, name=window_date)
                                betas[sector].append(params)
                                continue
                            
                            # Slice PC scores to match non-NaN y indices
                            row_pos = pca_rows.get_indexer(y_aligned.index)
                            Z_sub = PC_scores_k[row_pos, :]
                            
                            # Regress aligned non-NaN returns on sliced PC scores
                            params = _backproject_sector_coefficients(
                                y_aligned.values, Z_sub, V_k, X_mean,
                                feature_names=list(X_pruned.columns),
                                use_ridge=True,
                                alpha=alpha
                            )
                            params.name = window_date
                            betas[sector].append(params)
                    else:
                        # PCA failed, use intercept-only
                        for sector in sectors:
                            r_win = r[sector].loc[window]
                            params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                            betas[sector].append(params)
                else:
                    # All features pruned, use intercept-only
                    for sector in sectors:
                        r_win = r[sector].loc[window]
                        params = pd.Series({'const': float(r_win.mean())}, name=window_date)
                        betas[sector].append(params)
            except Exception:
                # If PCA fails, skip this window for all sectors
                pass
        else:
            # Standard Ridge path (no PCA): fit per sector with complete rows
            for sector in sectors:
                try:
                    # Get sector's returns for this window
                    r_win = r[sector].loc[window]
                    X_win = X_win_full
                    
                    # Align returns and signals to common non-NaN rows
                    combined = pd.concat([r_win, X_win], axis=1).dropna(how='any')
                    if combined.shape[0] < min_months:
                        continue
                    
                    y_vec = combined.iloc[:, 0].values
                    X_mat = combined.iloc[:, 1:].values
                    
                    # Fit Ridge
                    mdl = Ridge(alpha=alpha, fit_intercept=True)
                    mdl.fit(X_mat, y_vec)
                    
                    # Extract coefficients
                    col_names = ['const'] + list(X_win.columns)
                    coefs = [float(mdl.intercept_)] + [float(c) for c in mdl.coef_]
                    params = pd.Series(dict(zip(col_names, coefs)), name=window_date)
                    betas[sector].append(params)
                except Exception:
                    pass
    
    # Convert lists to DataFrames
    for sector in sectors:
        betas[sector] = pd.DataFrame(betas[sector]).sort_index() if betas[sector] else pd.DataFrame()
    
    return betas


def rolling_sectors_betas(monthly_rets: pd.DataFrame,
                          z_signals: pd.DataFrame,
                          lookback_months: int = 36,
                          min_months: int = 24,
                          assume_signals_already_lagged: bool = True):
    """
    Dispatcher: choose OLS or Ridge based on config.USE_RIDGE.
    Also reports PCA usage if enabled.
    """
    use_ridge = bool(getattr(config, "USE_RIDGE", False))
    use_pca = bool(getattr(config, "USE_PCA", False))
    
    # Build info message
    method = "Ridge" if use_ridge else "OLS"
    pca_info = ""
    if use_pca:
        var_target = float(getattr(config, "PCA_VAR_TARGET", 0.80))
        k_max = int(getattr(config, "PCA_K_MAX", 6))
        pca_info = f" + PCA (var_target={var_target}, k_max={k_max})"
    
    if use_ridge:
        alpha = float(getattr(config, "RIDGE_ALPHA", 5.0))
        print(f"[INFO] Rolling {method} betas (alpha={alpha}){pca_info}")
        return rolling_sectors_betas_ridge(
            monthly_rets, z_signals,
            lookback_months=lookback_months,
            min_months=min_months,
            alpha=alpha,
            assume_signals_already_lagged=assume_signals_already_lagged
        )
    else:
        print(f"[INFO] Rolling {method} betas{pca_info}")
        return rolling_sectors_betas_ols(
            monthly_rets, z_signals,
            lookback_months=lookback_months,
            min_months=min_months,
            assume_signals_already_lagged=assume_signals_already_lagged
        )


def sector_expected_returns(betas_dict: dict[str, pd.DataFrame],
                            z_signals: pd.DataFrame) -> pd.DataFrame:
    r"""
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