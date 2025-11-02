## Repository: Alternative-Investments-Sector-Rotation-Strategy

Short, actionable instructions to help an AI coding agent work productively in this repo.

1) Big-picture architecture
- This is a small Python backtest project that builds a monthly sector rotation strategy using ETF prices, a set of macro/market signals, rolling time-series regressions, and a simple rank-and-weight portfolio construction.
- Key components:
  - `main.py` — entrypoint: orchestrates data loading, signal processing, beta estimation, forecast generation, portfolio construction, simulation and plotting.
  - `data_loader.py` — file I/O helpers. Loads ETF price Excel files from `data/etf/`, signal files from `data/signals/`, and factor files from `data/factors/`.
  - `signal_processing.py` — resampling signals to month-end, z-scoring, monthly ETF return calculation and application of inception masks.
  - `exposure_model.py` — rolling sector-level OLS regressions that estimate betas of sector returns on standardized signals; produces per-sector expected returns.
  - `portfolio_construction.py` — ranking and long/short weights (configurable `LONG_SHORT_PAIRS`, optional inverse-vol weighting via `USE_VOL_TARGETING`).
  - `backtest.py` — applies lagged weights to next-month returns and (optionally) simple transaction cost handling.
  - `benchmark.py`, `utils/performance.py`, `utils/diagnostics.py`, `utils/plotting.py` — factor alignment, metrics and plotting helpers.

2) Where data lives and expected formats
- Data directory: `data/` with subfolders `etf/`, `signals/`, `factors/`.
- ETF files: `data/etf/<TICKER>.xlsx` with columns containing a Date and an adjusted close column (named `Adj Close`, `AdjClose` or `Close`). Parser returns DataFrames keyed by ticker via `load_sector_etf_prices()`.
- Signals: many files under `data/signals/` (examples in `data_loader.py`). Files may be daily or monthly; `monthly_signal_panel()` resamples heuristically to month-end and shifts by 1 month to avoid look-ahead.
- Fama-French + Mom factors: single Excel file configured in `config.py` as `FF5MOM_FILE`. Loader normalizes column names and converts percent -> decimal if necessary.

3) Run / dev commands
- Environment: project expects Python 3.11+ style packages listed in `requirements.txt` (pandas, numpy, statsmodels, matplotlib, openpyxl, ...). Use a virtualenv and install requirements.
  - Typical setup (bash):
    - python -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
- Run the strategy end-to-end:
    python main.py
  - `main.py` will write figures into `figs/` (created by `utils/diagnostics.py`) and print performance summaries.
- Quick checks: run individual modules from REPL or a short script to validate loading:
    from data_loader import load_sector_etf_prices; load_sector_etf_prices()

4) Project-specific conventions and patterns (do not change without checking callers)
- Time indexing and resampling: signals and prices are frequently resampled to month-end (`resample('ME')` or adding `MonthEnd`). Many functions assume month-end-aligned DatetimeIndex.
- Lagging to avoid lookahead: monthly signals are shifted by 1 month inside `monthly_signal_panel()` — most downstream code expects signals to be already lagged.
- Weight timing: weights computed at time t are applied to returns at t+1. `simulate_portfolio()` explicitly shifts weights by 1 before multiplying with returns.
- Inception mask: `apply_inception_mask()` sets returns to NaN before each ETF's inception date (month-end aware). Keep date handling consistent.
- Factor frequency and units: `load_ff5_mom_monthly()` will convert Excel percent columns into decimals if the sample mean is large (>0.5). Regression code assumes decimals (e.g., 0.01 for 1%).
- Missing data handling: functions commonly align indices via intersection and drop NaNs; when adding support for new signals/data, be explicit about alignment and NaN tolerance.

5) Important configuration knobs (see `config.py`)
- `BACKTEST_START`, `BACKTEST_END` — backtest window applied early in `main.py`.
- `SECTOR_TICKERS`, `ETF_INCEPTION` — canonical universe and per-ETF inception dates used by loaders and masks.
- `LONG_SHORT_PAIRS`, `USE_VOL_TARGETING`, `LOOKBACK_VOL_MONTHS`, `Z_WINDOW_MONTHS` — portfolio behavior and signal standardization.
- `TCOST_BPS` — transaction cost basis points applied in `backtest.py` (currently 0; `apply_tcosts` is basic and incomplete — be careful if you modify it).

6) Common edits an AI agent is likely to perform and where to update
- Adding a new signal: update `data/` (file), then `data_loader.load_signals()` if it has a nonstandard format; `signal_processing.monthly_signal_panel()` will include it automatically if the loader returns it in the signals dict.
- Add a new ETF: add file `data/etf/<TICKER>.xlsx` and append ticker to `SECTOR_TICKERS` in `config.py` and add inception date to `ETF_INCEPTION`.
- Change weighting scheme: modify `portfolio_construction.rank_and_weight_from_forecast()` — note it expects `forecast_df` and `monthly_rets` to share indices and columns; return value should be a DataFrame indexed by timestamps with columns=tickers.
- Improve transaction costs: `backtest.apply_tcosts()` is incomplete (it contains a bug/placeholder). If adding realistic tcosts, update `turnover()` and loop in `apply_tcosts()` to compute per-period costs; tests should validate Net vs Gross difference.

7) Tests, linting and quick verification
- There are no unit tests in the repo. Before major edits, run a quick syntax check (or small script) to import top-level modules:
    python -c "import data_loader,signal_processing,exposure_model,portfolio_construction,backtest,benchmark,utils.performance"
- Where you change public behavior, add small regression tests under `tests/` (pytest) that exercise: loading ETFS, monthly signal panel, simulate_portfolio on small synthetic data.

8) Examples & snippets (copy/paste friendly)
- Run full backtest: `python main.py` (will print summary table and save figures to `figs/`).
- Inspect a loaded signal:
    from data_loader import load_signals
    sigs = load_signals()
    list(sigs.keys())  # shows available names like 'VIX','OIL','COPPER'

9) Integration points and external dependencies
- Reads Excel files using `pandas.read_excel(..., engine='openpyxl')` — ensure `openpyxl` is installed.
- Stats regressions use `statsmodels` (OLS). Plotting uses `matplotlib` and `seaborn` (for heatmap). Large Excel files for FF5 factors are expected.

10) When to ask the user
- If adding or renaming a signal, confirm the canonical short name expected by `exposure_model` (these are columns in the z-score panel).
- If changing the ETF universe (adding/removing tickers) confirm whether portfolio sizing or mapping to data files needs updating.

If anything here is unclear or you want additional rules (for example stricter typing, automated tests, or CI commands), tell me what to add and I'll iterate.
