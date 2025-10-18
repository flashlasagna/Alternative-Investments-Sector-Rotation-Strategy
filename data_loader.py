import os
import pandas as pd
from config import ETF_DIR, SIGNALS_DIR, FACTORS_DIR, SECTOR_TICKERS, ETF_INCEPTION, REL_MOM_DIR


def _read_xlsx(path, sheet_name=0, **kwargs):
    """
    Returns a dict of DataFrames keyed by ticker with a DatetimeIndex,
    and at least a column 'AdjClose' (or 'TotalReturn' if you have it).
    """
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)

# ---------------- ETFs ----------------

def load_sector_etf_prices():
    """
    Loads each ETF file <TICKER>.xlsx with columns:
    Date, Open, High, Low, Close, Adj Close
    Returns dict[ticker] -> DataFrame(index=Date, ['AdjClose'])
    """
    data = {}
    for tkr in SECTOR_TICKERS:
        fp = os.path.join(ETF_DIR, f"{tkr}.xlsx")
        if not os.path.exists(fp):
            continue
        df = _read_xlsx(fp)
        # Standardize columns
        rename_map = {c: c.strip() for c in df.columns}
        df = df.rename(columns=rename_map)
        if "Date" not in df.columns:
            # assume first column is date if unnamed
            df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        if "Adj Close" in df.columns:
            px = df[["Adj Close"]].rename(columns={"Adj Close": "AdjClose"})
        elif "AdjClose" in df.columns:
            px = df[["AdjClose"]]
        else:
            px = df[["Close"]].rename(columns={"Close": "AdjClose"})
        data[tkr] = px.dropna()
    return data
# -------------- Relative Momentum (per ticker file) --------------
def load_relative_momentum_signals():
    """
      Loads daily relative momentum per ETF from files:
        data/signals/SPDR_RelMom/<TICKER>_RelMom.xlsx
      Each has two columns: Date, Value (or unnamed 2nd column)
      Returns wide DF indexed by Date with columns=tickers (if files exist).
      """
    frames = []
    for tkr in SECTOR_TICKERS:
        fp = os.path.join(REL_MOM_DIR, f"{tkr}_RelMom.xlsx")
        if not os.path.exists(fp):
            continue
        df = _read_xlsx(fp)
        if "Date" not in df.columns:
            # assume first column is date if unnamed
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: "tkr"})
        else:
            #name the non-data column as tkr
            valcol = [c for c in df.columns if c != "Date"][0]
            df = df.rename(columns={valcol: tkr})
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index() if hasattr(pd.DataFrame, "sort_index") else df.set_index("Date").sort_index()
        frames.append(df[[tkr]])
        if not frames:
            return None
        panel = pd.concat(frames, axis=1).sort_index()
    return panel

#Macro and Market Signals
def load_signals():
    """
      Loads all specified signal files and returns dict[name] -> DataFrame(index=Date).
      If file has two columns (Date, Value) it is read directly.
      Where multi-column, columns are mapped to canonical names below.
      """
    sig = {}

    #Consumer Sentiment Index (monthly)
    csi_fp = os.path.join(SIGNALS_DIR, "consumer_sentiment_index.xlsx")
    csi = _read_xlsx(csi_fp)
    csi = csi.rename(columns={csi.columns[0]: "Date"})
    csi["Date"] = pd.to_datetime(csi["Date"])
    csi = csi.set_index("Date").sort_index()
    #Map long-names to short names
    col_map ={}
    for c in csi.columns:
        cl = c.strip().lower()
        if "expectations" in cl:
            col_map[c] = "UMICH_EXP"
        else:
            col_map[c] = "UMICH_SENT"
    csi = csi.rename(columns=col_map)


