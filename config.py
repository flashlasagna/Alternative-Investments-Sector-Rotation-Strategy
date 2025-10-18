from datetime import date

DATA_DIR = "data"
ETF_DIR = f"{DATA_DIR}/etf"
SIGNALS_DIR = f"{DATA_DIR}/signals"
REL_MOM_DIR = f"{SIGNALS_DIR}/SPDR_RelMom"
FACTORS_DIR = f"{DATA_DIR}/factors"

SECTOR_TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",  # core
    "XLRE", "XLC"]

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
    "XLRE": "2015-10-08",
    "XLC": "2018-06-19",
}

BACKTEST_START = "2009-01-01"
BACKTEST_END = "2024-12-31"

REBAL_FREQ = "M"  # Monthly rebalancing

#Portfolio sizing
LONG_SHORT_PAIRS = 5  # Number of long/short pairs
USE_VOL_TARGETING = False
LOOBACK_VOL_MONTHS = 6

#signal normalization
Z_WINDOW_MONTHS = 24

#transaction costs
TRANSACTION_COST = 0  # As per directives

#Benchmark
FF5MOM_FILE = f"{FACTORS_DIR}/F-F_Research_Data_5_Factors_2x3_daily.csv"
FF_COLS = ["Mkt_RF","SMB","HML","RMW","CMA","RF","Mom"]

RANDOM_SEED = 42
