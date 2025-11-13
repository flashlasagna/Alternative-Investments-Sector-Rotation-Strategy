import os
import pandas as pd
from config import ETF_DIR, SIGNALS_DIR, FACTORS_DIR, SECTOR_TICKERS, ETF_INCEPTION, REL_MOM_DIR


def _read_xlsx(path, sheet_name=0, **kwargs):
    """
    Returns a dict of DataFrames keyed by ticker with a DatetimeIndex,
    and at least a column 'AdjClose' (or 'TotalReturn' if you have it).
    """
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", **kwargs)

def parse_dates(series):
    """Robust date parser for dd/mm/yyyy or dd.mm.yyyy formats."""
    return pd.to_datetime(series, dayfirst=True, errors="coerce")

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
        df["Date"] = parse_dates(df["Date"])
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
            df = df.rename(columns={df.columns[0]: "Date", df.columns[1]: tkr})
        else:
            valcol = [c for c in df.columns if c != "Date"][0]
            df = df.rename(columns={valcol: tkr})
        df["Date"] = parse_dates(df["Date"])
        df = df.set_index("Date").sort_index()
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

    # === Consumer Sentiment Index (UMICH Expectations only, monthly) ===
    csi_fp = os.path.join(SIGNALS_DIR, "ConsumerSentimentIndex.xlsx")
    csi = _read_xlsx(csi_fp)

    # Clean & parse
    csi = csi.rename(columns={csi.columns[0]: "Date"})
    csi["Date"] = parse_dates(csi["Date"])
    csi = csi.set_index("Date").sort_index()

    # Map and keep only Expectations column
    col_map = {}
    for c in csi.columns:
        cl = c.strip().lower()
        if "expectations" in cl:
            col_map[c] = "UMICH_EXP"
    csi = csi.rename(columns=col_map)

    # Only retain UMICH_EXP (drop headline sentiment)
    if "UMICH_EXP" in csi.columns:
        sig["UMICH_EXP"] = csi[["UMICH_EXP"]].dropna()

    # Copper & Gold prices (daily)
    cg_fp = os.path.join(SIGNALS_DIR, "Copper&Gold_Prices.xlsx")
    cg = _read_xlsx(cg_fp)
    cg = cg.rename(columns={cg.columns[0]: "Date"})
    cg["Date"] = parse_dates(cg["Date"])
    cg = cg.set_index("Date").sort_index()
    # Expect columns with these exact names:
    # 'S&P GSCI Copper Spot - PRICE INDEX', 'S&P GSCI Gold Spot - PRICE INDEX'
    if "S&P GSCI Copper Spot - PRICE INDEX" in cg.columns:
        sig["COPPER"] = cg[["S&P GSCI Copper Spot - PRICE INDEX"]].rename(
            columns={"S&P GSCI Copper Spot - PRICE INDEX": "COPPER"}
        ).dropna()
    if "S&P GSCI Gold Spot - PRICE INDEX" in cg.columns:
        sig["GOLD"] = cg[["S&P GSCI Gold Spot - PRICE INDEX"]].rename(
            columns={"S&P GSCI Gold Spot - PRICE INDEX": "GOLD"}
        ).dropna()


    # Credit spread HY - IG (daily) -> construct spread = HY - Corporate
    cs_fp = os.path.join(SIGNALS_DIR, "Credit Spread HY-IG.xlsx")
    cs = _read_xlsx(cs_fp)
    cs = cs.rename(columns={cs.columns[0]: "Date"})
    cs["Date"] = parse_dates(cs["Date"])
    cs = cs.setindex("Date").sortindex() if hasattr(pd.DataFrame, "sortindex") else cs.set_index("Date").sort_index()
    # columns expected: 'High Yield index', 'Corporate index'
    hy_col = None
    ig_col = None
    for c in cs.columns:
        cl = c.strip().lower()
        if "high yield" in cl:
            hy_col = c
        if "corporate" in cl:
            ig_col = c
    if hy_col and ig_col:
        tmp = pd.DataFrame(index=cs.index)
        tmp["HY_IG"] = cs[hy_col] - cs[ig_col]
        sig["HY_IG"] = tmp.dropna()

    # Fed Funds Reserves (daily)
    fed_fp = os.path.join(SIGNALS_DIR, "FedFundsReserves.xlsx")
    fed = _read_xlsx(fed_fp)
    fed = fed.rename(columns={fed.columns[0]: "Date"})
    fed["Date"] = parse_dates(fed["Date"])
    fed = fed.set_index("Date").sort_index()
    val = [c for c in fed.columns if c != "Date"][0]
    sig["FEDRES"] = fed[[val]].rename(columns={val: "FEDRES"}).dropna()

    #Oil Prices (daily)
    oil_fp = os.path.join(SIGNALS_DIR, "OilPrices.xlsx")
    oil = _read_xlsx(oil_fp)
    oil = oil.rename(columns={oil.columns[0]: "Date"})
    oil["Date"] = parse_dates(oil["Date"])
    oil = oil.set_index("Date").sort_index()
    val = [c for c in oil.columns if c != "Date"][0]
    sig["OIL"] = oil[[val]].rename(columns={val: "OIL"}).dropna()

    # PMI Manufacturing (monthly)
    pmi_fp = os.path.join(SIGNALS_DIR, "PMI_Manufacturing.xlsx")
    pmi = _read_xlsx(pmi_fp)
    pmi = pmi.rename(columns={pmi.columns[0]: "Date"})
    pmi["Date"] = parse_dates(pmi["Date"])
    pmi = pmi.set_index("Date").sort_index()
    val = [c for c in pmi.columns if c != "Date"][0]
    sig["PMI"] = pmi[[val]].rename(columns={val: "PMI"}).dropna()

    # 10y-2y spread (daily)
    spr_fp = os.path.join(SIGNALS_DIR, "T10Y2Y.xlsx")
    spr = _read_xlsx(spr_fp)
    spr = spr.rename(columns={spr.columns[0]: "Date"})
    spr["Date"] = parse_dates(spr["Date"])
    spr = spr.set_index("Date").sort_index()
    val = [c for c in spr.columns if c != "Date"][0]
    sig["Y10_2Y"] = spr[[val]].rename(columns={val: "Y10_2Y"}).dropna()

    # USD Index (daily)
    dxy_fp = os.path.join(SIGNALS_DIR, "USD Index - FRED.xlsx")
    dxy = _read_xlsx(dxy_fp)
    dxy = dxy.rename(columns={dxy.columns[0]: "Date"})
    dxy["Date"] = parse_dates(dxy["Date"])
    dxy = dxy.set_index("Date").sort_index()
    val = [c for c in dxy.columns if c != "Date"][0]
    sig["DXY"] = dxy[[val]].rename(columns={val: "DXY"}).dropna()

    # VIX (daily)
    vix_fp = os.path.join(SIGNALS_DIR, "VIX - FRED.xlsx")
    vix = _read_xlsx(vix_fp)
    vix = vix.rename(columns={vix.columns[0]: "Date"})
    vix["Date"] = parse_dates(vix["Date"])
    vix = vix.set_index("Date").sort_index()
    val = [c for c in vix.columns if c != "Date"][0]
    sig["VIX"] = vix[[val]].rename(columns={val: "VIX"}).dropna()

    #USREC NBER(daily)
    usrec_fp = os.path.join(SIGNALS_DIR, "NBER_Recession.xlsx")
    usrec = _read_xlsx(usrec_fp)
    usrec = usrec.rename(columns={usrec.columns[0]: "Date"})
    usrec["Date"] = parse_dates(usrec["Date"])
    usrec = usrec.set_index("Date").sort_index()
    val= [c for c in usrec.columns if c != "Date"][0]
    sig["USREC"] = usrec[[val]].rename(columns={val: "USREC"}).dropna()

    return sig

# -------------- Fama-French 5 Factors + Momentum --------------
def load_ff5_mom_monthly(path):
    xls = pd.ExcelFile(path, engine="openpyxl")
    # Prefer a sheet with "month" in the name; else use the first.
    sheet = next((s for s in xls.sheet_names if "month" in s.lower()), xls.sheet_names[0])

    df = _read_xlsx(path, sheet_name=sheet)
    df = df.rename(columns={df.columns[0]: "Date"})

    # Parse dates: accept YYYYMM or datetime-like
    if pd.api.types.is_numeric_dtype(df["Date"]):
        df["Date"] = df["Date"].astype(int).astype(str).str.zfill(6) + "01"
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    else:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index = df.index + pd.offsets.MonthEnd(0)

    # Standardize column names
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("mkt-rf","mkt_rf","mktrf","market-rf","market_rf","mkt_rf (%)","mkt minus rf","excess market return"):
            colmap[c] = "Mkt_RF"
        elif cl == "smb":
            colmap[c] = "SMB"
        elif cl == "hml":
            colmap[c] = "HML"
        elif cl == "rmw":
            colmap[c] = "RMW"
        elif cl == "cma":
            colmap[c] = "CMA"
        elif cl in ("rf","riskfree","risk_free","risk-free","risk free"):
            colmap[c] = "RF"
        elif cl in ("mom","umd","momentum"):
            colmap[c] = "Mom"
    df = df.rename(columns=colmap)

    # Convert percent -> decimal if needed
    num = df.select_dtypes(include="number")
    if not num.empty and num.abs().mean().mean() > 0.5:
        df[num.columns] = num / 100.0

    keep = [c for c in ["Mkt_RF","SMB","HML","RMW","CMA","RF","Mom"] if c in df.columns]
    return df[keep].dropna(how="all")



#  VIX Loading (for Crisis Detection)
def load_vix_monthly():
    """
    Load VIX data and resample to monthly (last business day of month).
    Returns pd.Series with monthly VIX values.
    """
    vix_fp = os.path.join(SIGNALS_DIR, "VIX - FRED.xlsx")
    vix = _read_xlsx(vix_fp)
    
    # Parse date column
    vix = vix.rename(columns={vix.columns[0]: "Date"})
    vix["Date"] = parse_dates(vix["Date"])
    vix = vix.set_index("Date").sort_index()
    
    # Get the first value column
    val_col = [c for c in vix.columns if c != "Date"][0]
    vix_series = vix[val_col].rename("VIX")
    
    # Resample to monthly (last value of month)
    vix_monthly = vix_series.resample("ME").last().dropna()
    
    return vix_monthly


# Credit Spread Loading (for Crisis Detection) 
def load_credit_spread_monthly():
    """
    Load credit spread (HY - IG) data and resample to monthly.
    Returns pd.Series with monthly credit spread values (in basis points).
    """
    cs_fp = os.path.join(SIGNALS_DIR, "Credit Spread HY-IG.xlsx")
    cs = _read_xlsx(cs_fp)
    
    # Parse date column
    cs = cs.rename(columns={cs.columns[0]: "Date"})
    cs["Date"] = parse_dates(cs["Date"])
    cs = cs.set_index("Date").sort_index()
    
    # Find HY and IG columns
    hy_col = None
    ig_col = None
    for c in cs.columns:
        cl = c.strip().lower()
        if "high yield" in cl:
            hy_col = c
        if "corporate" in cl:
            ig_col = c
    
    if hy_col and ig_col:
        # Calculate spread (HY - IG, in basis points)
        cs_spread = cs[hy_col] - cs[ig_col]
        cs_monthly = cs_spread.resample("ME").last().dropna()
        return cs_monthly.rename("Credit_Spread")
    else:
        raise ValueError(f"Could not find HY and IG columns in {cs_fp}")


