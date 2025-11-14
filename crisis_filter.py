# 3 Core Functions:

# rolling_percentile() - Trailing percentile (no look-ahead)

# compute_crisis_flag()  =>Detects when BOTH VIX + credit spread exceed 80th percentile
# Hysteresis: Min 2 months to enter, 2 months to exit (prevents flapping)
# Returns boolean series: True = crisis


# apply_crisis_reduction() - Scale down positions
# - Multiplies weights by 0.5 during crisis
# - Preserves long/short ratio (maintains neutrality)

# get_crisis_info() =>  Shows total crisis months, percentage, first/last crisis dates


# IMPORTANT =>  IN  PORTFOLIO_CONSTRUCTION.PY  we have Updated scale_to_target_vol():

#Added crisis_flag parameter
#When crisis=True, caps vol scaler at 1.0
#Prevents undoing the crisis reduction 


import numpy as np
import pandas as pd


def _rolling_percentile(x: pd.Series, window: int, p: float) -> pd.Series:
    """
    Compute trailing percentile using only past data (no look-ahead).
    
    Args:
        x: Time series
        window: Lookback window in periods
        p: Percentile (0.80 = 80th percentile)
    
    Returns:
        pd.Series of percentiles at each timestamp
    """
    vals = []
    arr = x.values
    
    for i in range(len(arr)):
        lo = max(0, i - window)
        hist = arr[lo:i]  # up to t-1, NOT including current t
        
        if len(hist) == 0 or np.all(np.isnan(hist)):
            vals.append(np.nan)
        else:
            vals.append(np.nanpercentile(hist, p * 100))
    
    return pd.Series(vals, index=x.index)


def compute_crisis_flag(vix: pd.Series,
                        credit_spread: pd.Series,
                        window_months: int = 120,
                        p_vix: float = 0.7,
                        p_cs: float = 0.7,
                        logic: str = "AND",
                        min_on: int = 1,
                        min_off: int = 3) -> pd.Series:
    """
    Detect market crisis periods using trailing percentile thresholds.
    
    Crisis = True when BOTH VIX AND credit spread exceed their 80th percentile
    (or configured thresholds).
    
    Hysteresis prevents flapping: need min_on=2 consecutive months to ENTER crisis,
    and min_off=2 consecutive months to EXIT crisis.
    
    Args:
        vix: Monthly VIX series
        credit_spread: Monthly credit spread series (HY-IG, basis points)
        window_months: Lookback for percentile calculation (default 10 years)
        p_vix: VIX percentile threshold (0.80 = 80th percentile)
        p_cs: Credit spread percentile threshold (0.80 = 80th percentile)
        logic: "AND" (both high) or "OR" (either high)
        min_on: Months above threshold before entering crisis
        min_off: Months below threshold before exiting crisis
    
    Returns:
        pd.Series (bool) with True = crisis period
    
    Example:
        >>> vix = pd.Series([10, 15, 20, 25, 30, 35, 32, 28, 25, 20, ...])
        >>> cs = pd.Series([50, 60, 100, 150, 200, 250, 240, 180, 120, ...])
        >>> crisis = compute_crisis_flag(vix, cs)
        # Returns series with True during high stress periods
    """
    # Compute trailing percentile thresholds
    vix_thr = _rolling_percentile(vix, window_months, p_vix)
    cs_thr = _rolling_percentile(credit_spread, window_months, p_cs)
    
    # Detect raw stress condition (VIX AND spread both high)
    if logic == "AND":
        raw = (vix > vix_thr) & (credit_spread > cs_thr)
    else:  # "OR"
        raw = (vix > vix_thr) | (credit_spread > cs_thr)
    
    # Apply hysteresis to avoid flapping
    # Need min_on consecutive True to ENTER crisis
    # Need min_off consecutive False to EXIT crisis
    crisis = []
    state = False  # Not in crisis initially
    on_cnt = 0     # Count of consecutive stress signals
    off_cnt = 0    # Count of consecutive calm signals
    
    for t, flag in raw.fillna(False).items():
        if state:
            # Currently IN crisis
            if flag:
                # Still stressed: increment on counter
                on_cnt += 1
                off_cnt = 0
            else:
                # Stress lifted: increment off counter
                off_cnt += 1
                
                # Exit crisis if we've been calm for min_off months
                if off_cnt >= min_off:
                    state = False
                    on_cnt = off_cnt = 0
        else:
            # Currently NOT in crisis
            if flag:
                # Stress detected: increment on counter
                on_cnt += 1
                
                # Enter crisis if we've been stressed for min_on months
                if on_cnt >= min_on:
                    state = True
                    on_cnt = off_cnt = 0
            else:
                # Still calm: increment off counter
                off_cnt += 1
                on_cnt = 0
        
        crisis.append(state)
    
    return pd.Series(crisis, index=raw.index, name="crisis_flag")


def apply_crisis_reduction(weights: pd.DataFrame,
                           crisis_flag: pd.Series,
                           reduction: float = 0.50) -> pd.DataFrame:
    """
    Scale down portfolio positions during crisis periods.
    
    Multiplies weights by (1 - reduction) when crisis_flag=True.
    
    Example: reduction=0.50 means positions are cut to 50% during crisis
    (half of gross long, half of gross short).
    
    Args:
        weights: Portfolio weights DataFrame (rows=dates, cols=sectors)
        crisis_flag: Boolean series with True=crisis period
        reduction: Reduction factor (0.50 = 50% reduction)
    
    Returns:
        Adjusted weights with crisis scaling applied
    
    Example:
        >>> weights = pd.DataFrame({'XLK': [0.1, 0.1], 'XLV': [0.1, 0.1]})
        >>> crisis = pd.Series([False, True], index=weights.index)
        >>> reduced = apply_crisis_reduction(weights, crisis, reduction=0.50)
        # First row unchanged, second row scaled to 50%
    """
    # Create multiplier: 1.0 in normal times, (1-reduction) during crisis
    # Example: reduction=0.50 → multiplier = 0.50 during crisis
    m = (~crisis_flag).astype(float) + crisis_flag.astype(float) * (1.0 - reduction)
    
    # Align with weights index (fill any missing dates)
    m = m.reindex(weights.index).fillna(1.0)
    
    # Apply multiplier to all positions
    return (weights.T * m).T


def get_crisis_info(crisis_flag: pd.Series) -> dict:
    """
    Provide diagnostics on crisis periods.
    
    Args:
        crisis_flag: Boolean series of crisis indicators
    
    Returns:
        Dictionary with crisis statistics
    """
    crisis_periods = crisis_flag[crisis_flag].index
    normal_periods = crisis_flag[~crisis_flag].index
    
    info = {
        'total_months': len(crisis_flag),
        'crisis_months': len(crisis_periods),
        'normal_months': len(normal_periods),
        'crisis_pct': len(crisis_periods) / len(crisis_flag) * 100,
        'first_crisis': crisis_periods.min() if len(crisis_periods) > 0 else None,
        'last_crisis': crisis_periods.max() if len(crisis_periods) > 0 else None,
    }
    
    return info
