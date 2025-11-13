from datetime import date

# ---------- DATA PATHS ----------
DATA_DIR = "data"
ETF_DIR = f"{DATA_DIR}/etf"
SIGNALS_DIR = f"{DATA_DIR}/signals"
REL_MOM_DIR = f"{SIGNALS_DIR}/SPDR_RelMom"
FACTORS_DIR = f"{DATA_DIR}/factors"

# ---------- UNIVERSE ----------
SECTOR_TICKERS = [
    "XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY",  # core
    "XLRE",  # 2015-10-08
    "XLC",   # 2018-06-19
]

ETF_INCEPTION = {
    "XLB": "1998-12-22",
    "XLE": "1998-12-16",
    "XLF": "1998-12-22",
    "XLI": "1998-12-22",
    "XLK": "1998-12-16",
    "XLP": "1998-12-22",
    "XLU": "1998-12-22",
    "XLV": "1998-12-16",
    "XLY": "1998-12-22",
    "XLRE":"2015-10-08",
    "XLC": "2018-06-19",
    }
# BACKTEST 
BACKTEST_START = "2009-01-01"
BACKTEST_END   = "2024-12-31"
REBAL_FREQ = "M"  # monthly

# PORTFOLIO 
LONG_SHORT_PAIRS = 5          # long top N, short bottom N
USE_VOL_TARGETING = False     # inverse-vol weighting inside L/S buckets
LOOKBACK_VOL_MONTHS = 6
TCOST_BPS = 0                 # you set to 0

# SIGNALS 
Z_WINDOW_MONTHS = 24  # z-score rolling window

# FACTORS 

LOOKBACK_BETA_MONTHS = 36
MIN_BETA_MONTHS = 24
FF5MOM_FILE = f"{FACTORS_DIR}/F-F_Research_Data_5_Factors_2x3.xlsx"
# You mentioned: ["Mkt_RF","SMB","HML","RMW","CMA","RF","Mom"]
FF_EXPECTED_COLS = ["Mkt_RF","SMB","HML","RMW","CMA","RF","Mom"]

SEED = 42

# CRISIS FILTER (PHASE 2)
CRISIS_VIX_PERCENTILE = 0.9          # VIX > 90th percentile
CRISIS_CS_PERCENTILE = 0.9           # Credit spread > 90th percentile
CRISIS_WINDOW_MONTHS = 120          # 10-year lookback for percentiles
CRISIS_LOGIC = "AND"                  # Both VIX AND spread must be high
CRISIS_MIN_ON = 1                     # Months above threshold to ENTER crisis
CRISIS_MIN_OFF = 2                    # Months below threshold to EXIT crisis
CRISIS_REDUCTION = 0.5               # Reduce positions by 30% during crisis


# TRANSFORM SETTINGS(signal_processing.py)
# Columns to (safely) log-transform (must be strictly positive)
LOG_COLS = ["COPPER", "GOLD", "OIL", "DXY", "VIX", "FEDRES", "CG_RATIO"]

# Columns to add 1m differences (Δ1)
DIFF1_COLS = ["UMICH_EXP", "PMI", "Y10_2Y", "HY_IG", "OIL", "DXY", "VIX", "FEDRES", "CG_RATIO"]

# Columns to add 3m differences (Δ3) — optional, useful for cycle turns
DIFF3_COLS = ["UMICH_EXP", "PMI", "Y10_2Y", "HY_IG", "OIL", "DXY", "VIX", "FEDRES", "CG_RATIO"]

# If True, also include log(CG_RATIO) (equivalent to log(COPPER) - log(GOLD))
ADD_LOG_CG = True



# RIDGE REGRESSION (exposure_model.py)
# Use Ridge regression instead of OLS for rolling betas

USE_RIDGE = True # Set to True to enable Ridge; False uses OLS
RIDGE_ALPHA = 5  # Regularization strength (higher = more shrinkage)



# UNCOMMENT THIS JUST IF YOU WANNA TRY PCA 

# PCA BACK-PROJECTION (exposure_model.py)
# Use PCA to reduce dimensionality and back-project to original features
# USE_PCA = False  # Set to True to enable PCA back-projection
# PCA_VAR_TARGET = 0.90  # Target explained variance (e.g., 0.80 = 80%)
# PCA_K_MAX = 12 # Maximum number of principal components (hard cap)
# PCA_PREPRUNE_THRESH = 0.98  # Pre-prune near-duplicate features with |ρ| > threshold