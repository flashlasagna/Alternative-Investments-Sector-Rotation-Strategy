# Macro-Driven Sector Rotation Strategy

A production-ready, long/short sector rotation strategy that trades US sector ETFs (SPDR Select Sector SPDRs) using macroeconomic signals and rolling factor exposures. Built for the **Advanced Data Analytics** course at HEC Lausanne.

---

## 📊 Strategy Overview

### Core Concept
This strategy dynamically rotates between US equity sectors (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, XLC) based on predicted returns from macro signals. It maintains a market-neutral long/short portfolio with crisis-aware risk management.

### Key Features

**Signal Processing**
- 9 macro signals: VIX, credit spreads (HY-IG), yield curve (10Y-2Y), PMI, consumer sentiment, copper/gold ratio, oil prices, USD index, Fed reserves
- Engineered features: log transforms, 1-month & 3-month differences, regime dummies
- Availability-lagged (1-month lag) to prevent look-ahead bias
- Rolling z-score standardization (24-month window)

**Exposure Model**
- Rolling sector-level factor exposures using Ridge regression (L2 regularization)
- 36-month lookback, 24-month minimum for estimation
- Per-sector forecasts: $`\hat{r}_{i, t+1} = \alpha_{i, t} + \beta_{i, t}^\top z_t`$
- Optional PCA dimensionality reduction (commented out, configurable)

**Portfolio Construction**
1. **Hysteresis-based ranking**: Reduces turnover by 40-60% vs. naive ranking
   - Entry/exit thresholds with dwell-time friction
   - Universe-aware (handles 9-11 sectors dynamically)
   - Maintains minimum bucket sizes for neutrality
2. **Position caps**: Water-filling algorithm ensures no single sector exceeds limits
   - Long: 25% max per sector, 20% gross target
   - Short: 15% max per sector, 20% gross target
3. **Volatility targeting**: 12% annualized target with 3-month smoothing (trailing only)
4. **Crisis filter**: 50% position reduction when both VIX & credit spreads breach 70th percentile

**Risk Management**
- Market-neutral (20% gross long / 20% gross short)
- No look-ahead bias (trailing windows only)
- Crisis detection with hysteresis (1-month entry, 3-month exit)
- Volatility scaler capped at 1.0× during crises

---

## 🏆 Performance Summary

**Backtest Period**: March 2010 – December 2024 (175 months)

| Metric | Value |
|--------|-------|
| **Annualized Return** | 3.17% |
| **Annualized Volatility** | 4.78% |
| **Sharpe Ratio** | 0.7082 |
| **Maximum Drawdown** | -6.72% |
| **Sortino Ratio** | 1.65 |
| **Hit Ratio** | 46.88% |
| **FF5+Mom Alpha** | 3.17% (t=1.98) |
| **Market Beta** | -0.0518 (near-zero) |

**Crisis Performance**: Strategy reduced positions during 2020 COVID crash and maintained drawdowns below 12% throughout the sample.

---

## 📁 Repository Structure

```
├── main.py                      # Main backtest pipeline
├── config.py                    # All configuration parameters
├── requirements.txt             # Python dependencies
│
├── data_loader.py              # ETF prices, signals, factors loading
├── signal_processing.py        # Feature engineering, z-scoring
├── exposure_model.py           # Rolling Ridge regressions, forecasts
├── portfolio_construction.py   # Hysteresis ranking, caps, vol targeting
├── crisis_filter.py            # VIX/spread crisis detection
├── backtest.py                 # Return simulation, transaction costs
├── benchmark.py                # Fama-French factor regression
│
├── sector_analysis.py          # Sector-level diagnostics script
│
├── test_ridge_alphas.py        # Regularization strenght test
│
├── utils/
│   ├── performance.py          # Performance metrics
│   └── diagnostics.py          # Plotting functions
│
├── data/
│   ├── etf/                    # Sector ETF price data (.xlsx)
│   ├── signals/                # Macro signal files (.xlsx)
│   └── factors/                # Fama-French 5-factor + Momentum
│
├── figs/                       # Generated diagnostic plots (PDF)
└── UPDATED_IMPLEMENTATION.md   # Technical implementation notes
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.10+
pip install -r requirements.txt
```

**Dependencies:**
- pandas >= 2.2
- numpy >= 1.26
- scipy >= 1.12
- statsmodels >= 0.14
- openpyxl >= 3.1
- matplotlib >= 3.8
- scikit-learn (for Ridge/PCA)

### Run Backtest
```bash
python main.py
```

**Outputs:**
- Console: Performance metrics, alpha stats, crisis diagnostics
- `figs/`: 15+ diagnostic plots (equity curves, drawdowns, IC, turnover, etc.)

### Run Sector Analysis
```bash
python sector_analysis.py
```
Generates sector-level activity plots (position frequency, weight evolution).

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Portfolio Parameters
```python
BACKTEST_START = "2009-01-01"
BACKTEST_END   = "2024-12-31"

# Hysteresis thresholds
ENTRY_THRESHOLD = 3      # Enter long if rank ≤ 3
EXIT_THRESHOLD = 6       # Exit long if rank ≥ 6
DWELL_MIN = 2            # Min months in bucket (friction)
MIN_BUCKET_SIZE = 4      # Min longs/shorts for neutrality

# Position limits
L_TARGET = 0.20          # 20% gross long
S_TARGET = 0.20          # 20% gross short
MAX_LONG_SINGLE = 0.25   # 25% max per sector long
MAX_SHORT_SINGLE = 0.15  # 15% max per sector short

# Volatility targeting
VOL_TARGET = 0.12        # 12% annualized
LOOKBACK_VOL = 12        # 12-month rolling window
SMOOTHING_WINDOW = 3     # 3-month smoothing
```

### Signal Settings
```python
Z_WINDOW_MONTHS = 24     # Z-score lookback

# Feature engineering
LOG_COLS = ["COPPER", "GOLD", "OIL", "DXY", "VIX", "FEDRES", "CG_RATIO"]
DIFF1_COLS = ["UMICH_EXP", "PMI", "Y10_2Y", "HY_IG", ...]
DIFF3_COLS = [...]       # 3-month acceleration
```

### Exposure Model
```python
USE_RIDGE = True         # Ridge vs OLS
RIDGE_ALPHA = 4.0        # Regularization strength
LOOKBACK_BETA_MONTHS = 36
MIN_BETA_MONTHS = 24
```

### Crisis Filter
```python
CRISIS_VIX_PERCENTILE = 0.7     # 70th percentile threshold
CRISIS_CS_PERCENTILE = 0.7      # Credit spread threshold
CRISIS_WINDOW_MONTHS = 120      # 10-year lookback
CRISIS_LOGIC = "AND"            # Both must trigger
CRISIS_MIN_ON = 1               # Months to enter crisis
CRISIS_MIN_OFF = 3              # Months to exit crisis
CRISIS_REDUCTION = 0.5          # 50% position cut
```

---

## 📈 Key Improvements Over Naive Approaches

### 1. Hysteresis Ranking (`portfolio_construction.py`)
**Problem**: Naive rank-based strategies flip sectors at boundaries → high turnover, slippage.

**Solution**: Entry/exit thresholds with dwell-time friction.
- Entry long if rank ≤ 3, exit only if rank ≥ 6
- Symmetric for shorts (universe-aware)
- **Result**: 40-60% turnover reduction, -11.8% max DD (vs. -28.7% before)

### 2. Water-Filling Caps (`apply_position_caps_and_renormalize`)
**Problem**: One-shot clipping violates caps or misses target exposure.

**Solution**: Iterative water-filling algorithm.
- Distributes overflow to non-capped positions
- Guarantees no cap violations AND hits exact gross targets
- **Result**: Prevented 2020-style concentration blowups

### 3. Trailing-Only Vol Scaling (`scale_to_target_vol`)
**Problem**: `center=True` in smoothing = look-ahead bias → inflated backtest Sharpe.

**Solution**: Removed centering; crisis scaler cap at 1.0×.
- Uses only past data for vol estimation
- Prevents vol targeting from undoing crisis reduction
- **Result**: Live-tradeable, honest performance metrics

### 4. Crisis-Aware Risk (`crisis_filter.py`)
**Problem**: Vol targeting can upscale during stress (bad timing).

**Solution**: Dual-signal crisis detection (VIX + credit spreads) with hysteresis.
- 50% position reduction during crises
- Vol scaler capped at 1.0× to preserve cut
- **Result**: Max DD contained to -11.8% during COVID

---

## 🔬 Diagnostics & Validation

### Generated Plots (see `figs/`)
1. **Equity Curves** (`equity_curve.pdf`): Gross vs. net returns
2. **Drawdown** (`drawdown.pdf`): Underwater chart
3. **Rolling Sharpe** (`rolling_sharpe_w24.pdf`): 24-month rolling
4. **Rolling Alpha** (`rolling_alpha_w24.pdf`): FF5+Mom alpha over time
5. **Factor Exposures** (`factor_exposures_ff5_mom.pdf`): Bar chart of betas
6. **Signal Correlation** (`signal_corr_heatmap.pdf`): Macro signal correlations
7. **Cross-Sectional IC** (`cs_ic.pdf`): Forecast vs. realized returns
8. **Top-Bottom Spread** (`tb_spread_n3.pdf`): Long top-3 / short bottom-3
9. **Vol Scaling** (`vol_scaling_series_vol_targeting.pdf`): Realized scalers
10. **Sector Weights** (`sector_weights_long_short_stacked.pdf`): Weight evolution
11. **Turnover** (`turnover_timeseries.pdf`, `turnover_hist.pdf`): Monthly turnover
12. **Subperiod Sharpe** (`subperiod_sharpe.pdf`): Performance by regime
13. **Return Distribution** (`return_distribution.pdf`): Histogram + KDE

### Console Output
- First/last trading dates per data stage (signals → betas → forecasts → weights)
- Crisis period diagnostics (total months, % in crisis, dates)
- FF5+Mom regression summary (HAC t-stats)
- Extended metrics (Sortino, hit ratio, skew, kurtosis)

---

## 📚 Data Requirements

### ETF Prices (`data/etf/`)
- Files: `XLB.xlsx`, `XLE.xlsx`, ..., `XLC.xlsx`
- Columns: `Date, Open, High, Low, Close, Adj Close`
- Format: Daily, any date format (parsed with `dayfirst=True`)

### Signals (`data/signals/`)
| File | Frequency | Column(s) |
|------|-----------|-----------|
| `VIX - FRED.xlsx` | Daily | VIX |
| `Credit Spread HY-IG.xlsx` | Daily | High Yield index, Corporate index |
| `T10Y2Y.xlsx` | Daily | 10Y-2Y spread |
| `PMI_Manufacturing.xlsx` | Monthly | PMI |
| `ConsumerSentimentIndex.xlsx` | Monthly | Expectations |
| `Copper&Gold_Prices.xlsx` | Daily | S&P GSCI Copper/Gold Spot |
| `OilPrices.xlsx` | Daily | Oil price |
| `USD Index - FRED.xlsx` | Daily | USD Index (DXY) |
| `FedFundsReserves.xlsx` | Daily | Fed reserves |
| `NBER_Recession.xlsx` | Daily | USREC (0/1) |

### Factors (`data/factors/`)
- `F-F_Research_Data_5_Factors_2x3.xlsx`: FF5 + Momentum (monthly)
- Columns: `Mkt-RF, SMB, HML, RMW, CMA, RF, Mom`

---

## 🧪 Advanced Features (Optional)

### PCA Dimensionality Reduction
Uncomment in `config.py` and `exposure_model.py`:
```python
USE_PCA = True
PCA_VAR_TARGET = 0.90        # Retain 90% variance
PCA_K_MAX = 12               # Max 12 principal components
PCA_PREPRUNE_THRESH = 0.98   # Drop near-duplicates
```
Back-projects coefficients to original features after fitting on PCs.

### Walk-Forward State Persistence
`rank_and_weight_from_forecast_hysteresis()` returns `(weights, state_dict)`:
```python
weights, state = rank_and_weight_from_forecast_hysteresis(
    forecast_df, monthly_rets, state_dict=previous_state
)
```
Use for production walk-forward without resetting hysteresis.

---

## 🛠️ Troubleshooting

### Common Issues

**"No trading activity detected"**
- Check `ETF_INCEPTION` dates in `config.py`
- Verify signal availability: `first_valid()` diagnostics in `main.py`

**High turnover (>0.6/month)**
- Increase `DWELL_MIN` (e.g., 3 months)
- Widen hysteresis gap: `EXIT_THRESHOLD = 7`

**Alpha t-stat < 1.5**
- Normal for macro strategies; t=1.91 is economically significant
- Check IC plot: mean IC > 0.05 is good

**Negative Sharpe in subperiod**
- Expected during regime shifts (e.g., 2015-2017 low vol)
- Crisis filter may keep portfolio flat if signals weak

---

## 📝 Citation

```bibtex
@misc{sector_rotation_2024,
  authors = {Ruben Mimouni, Federica Keni, Jacopo Sinigaglia, Nour Kabbara},
  title = {Macro-Driven Sector Rotation Strategy},
  year = {2025},
  institution = {HEC Lausanne - Applied Investments,
  supervisor = {Pr. Amit Goyal}
}
```

---

## 🤝 Contributing

This is an academic project. For questions or suggestions:
- Open an issue with diagnostic plots attached
- Provide `main.py` console output
- Specify Python/package versions

---

## 📜 License

Academic use only. Not intended for live trading without proper risk disclosure and compliance review.

---

## 🙏 Acknowledgments

- **Select Sector SPDR** - Sector ETF data
- **FRED Economic Data** - Macro signals
- **Datastream** - Macro signals
- **Kenneth French Data Library** - Factor returns
