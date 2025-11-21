"""
Sector Analysis Module
======================
Track which sectors are active and their weight evolution over time.

Usage:
    Before running this part you need to run the main file 
    python sector_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import (
    BACKTEST_START, BACKTEST_END, ETF_INCEPTION, LOOKBACK_VOL_MONTHS, 
    LOOKBACK_BETA_MONTHS, MIN_BETA_MONTHS,
    CRISIS_VIX_PERCENTILE, CRISIS_CS_PERCENTILE, CRISIS_WINDOW_MONTHS, 
    CRISIS_LOGIC, CRISIS_MIN_ON, CRISIS_MIN_OFF, CRISIS_REDUCTION
)
from data_loader import (
    load_sector_etf_prices, load_signals, 
    load_vix_monthly, load_credit_spread_monthly
)
from signal_processing import (
    etf_monthly_returns, apply_inception_mask, monthly_signal_panel, 
    zscore_signals
)
from exposure_model import rolling_sectors_betas, sector_expected_returns
from portfolio_construction import (
    rank_and_weight_from_forecast_hysteresis, 
    apply_position_caps_and_renormalize, scale_to_target_vol
)
from crisis_filter import compute_crisis_flag, apply_crisis_reduction


def plot_sector_activity(weights):
    """Show when each sector is ACTIVE (has non-zero position) vs INACTIVE."""
    print("\nPlotting sector activity (active/inactive)...")
    
    # Filter to March 2010 - December 2024
    start_date = "2010-03-01"
    end_date = "2024-12-31"
    W = weights.loc[start_date:end_date].copy().fillna(0.0)
    
    # Binary: 1 if sector has any position, 0 otherwise
    active = (W != 0).astype(int)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create a color map: green for active, light gray for inactive
    colors = np.zeros((len(active.columns), len(active)))
    for i, sector in enumerate(active.columns):
        colors[i] = active[sector].values
    
    # Plot as imshow with custom colors
    im = ax.imshow(colors, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_yticks(range(len(active.columns)))
    ax.set_yticklabels(active.columns, fontsize=11)
    
    # Set x-axis to show dates
    x_ticks = np.linspace(0, len(active) - 1, 15, dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([active.index[i].strftime('%Y-%m') for i in x_ticks], rotation=45, fontsize=10)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sector', fontsize=12, fontweight='bold')
    ax.set_title('Sector Activity Timeline (Green = Active Position, Red = Inactive)\nMarch 2010 - December 2024', 
                 fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Active/Inactive')
    plt.tight_layout()
    plt.savefig('figs/sector_activity_timeline.pdf', bbox_inches='tight', dpi=300)
    print("✓ Saved: figs/sector_activity_timeline.pdf")
    plt.show()


def plot_sector_position_counts(weights):
    """Show histograms for each sector: months in LONG, SHORT, or NEUTRAL."""
    print("\nPlotting sector position histograms...")
    
    # Filter to March 2010 - December 2024
    start_date = "2010-03-01"
    end_date = "2024-12-31"
    W = weights.loc[start_date:end_date].copy().fillna(0.0)
    
    # Count position types for each sector
    n_sectors = len(W.columns)
    n_cols = 4
    n_rows = (n_sectors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for idx, sector in enumerate(W.columns):
        ax = axes[idx]
        
        w = W[sector]
        long_count = (w > 0).sum()
        short_count = (w < 0).sum()
        neutral_count = (w == 0).sum()
        
        # Create histogram
        positions = ['Long', 'Short', 'Neutral']
        counts = [long_count, short_count, neutral_count]
        colors = ['green', 'red', 'lightgray']
        
        bars = ax.bar(positions, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Number of Months', fontsize=10)
        ax.set_title(f'{sector}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Remove extra subplots
    for idx in range(n_sectors, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('Sector Position Frequency - Months in Each Status\nMarch 2010 - December 2024', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('figs/sector_position_frequency.pdf', bbox_inches='tight', dpi=300)
    print("✓ Saved: figs/sector_position_frequency.pdf")
    plt.show()


def plot_sector_weights_evolution(weights):
    """Track individual sector weight evolution over time."""
    print("\nPlotting sector weight evolution...")
    
    # Filter to March 2010 - December 2024
    start_date = "2010-03-01"
    end_date = "2024-12-31"
    W = weights.loc[start_date:end_date].copy().fillna(0.0)
    
    # Create a plot for each sector separately
    n_sectors = len(W.columns)
    n_cols = 4
    n_rows = (n_sectors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows), sharex=True)
    axes = axes.flatten()  # Flatten in case of 2D array
    
    for idx, sector in enumerate(W.columns):
        ax = axes[idx]
        
        w = W[sector]
        
        # Color bars: green for long, red for short
        colors = ['green' if x > 0 else 'red' for x in w.values]
        ax.bar(w.index, w.values, color=colors, alpha=0.7, width=25)
        
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(f'{sector}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weight', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis ticks to show every 2 years
        years = pd.date_range(start=start_date, end=end_date, freq='2YS')
        ax.set_xticks(years)
        ax.set_xticklabels([year.year for year in years], rotation=45, fontsize=8)
    
    # Remove extra subplots
    for idx in range(n_sectors, len(axes)):
        fig.delaxes(axes[idx])
    
    # Set x-label for bottom plots
    for idx in range(len(axes) - n_cols, len(axes)):
        if idx < len(axes) and idx < n_sectors:
            axes[idx].set_xlabel('Date', fontsize=10)
    
    fig.suptitle('Sector Weight Evolution Over Time (Green=Long, Red=Short)\nMarch 2010 - December 2024', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('figs/sector_weights_evolution.pdf', bbox_inches='tight', dpi=300)
    print("✓ Saved: figs/sector_weights_evolution.pdf")
    plt.show()


def print_sector_summary(weights):
    """Print tables showing sector activity and statistics."""
    
    print("\n" + "="*80)
    print("SECTOR PORTFOLIO SUMMARY (March 2010 - December 2024)")
    print("="*80)
    
    # Filter to March 2010 - December 2024
    start_date = "2010-03-01"
    end_date = "2024-12-31"
    W = weights.loc[start_date:end_date].copy().fillna(0.0)
    
    # 1. Sector statistics
    print("\nSECTOR STATISTICS:")
    print("-" * 80)
    
    stats = pd.DataFrame({
        'Avg Weight': W.mean(),
        'Max Weight': W.max(),
        'Min Weight': W.min(),
        'Std Dev': W.std(),
        'Months Active': (W != 0).sum(),
        'Activity %': ((W != 0).sum() / len(W) * 100).round(1)
    })
    print(stats.round(4))
    
    # 2. Most/Least traded sectors
    print("\nSECTOR ACTIVITY RANKING:")
    print("-" * 80)
    
    activity = (W != 0).sum().sort_values(ascending=False)
    print("\nMost Active Sectors (months in portfolio):")
    for sector, months in activity.head(5).items():
        print(f"  {sector:6s}: {months:3d} months ({months/len(W)*100:.1f}%)")
    
    print("\nLeast Active Sectors (months in portfolio):")
    for sector, months in activity.tail(5).items():
        print(f"  {sector:6s}: {months:3d} months ({months/len(W)*100:.1f}%)")


def main():
    """Main sector analysis pipeline."""
    
    print("="*80)
    print("SECTOR ANALYSIS - TRACKING PORTFOLIO EVOLUTION")
    print("Period: March 2010 - December 2024")
    print("="*80)
    
    # Load data
    print("\n Loading data...")
    etf_px = load_sector_etf_prices()
    sigs_raw = load_signals()
    vix_monthly = load_vix_monthly()
    credit_spread_monthly = load_credit_spread_monthly()

    print("Computing returns and signals...")
    r_full = etf_monthly_returns(etf_px)
    r_full = apply_inception_mask(r_full, ETF_INCEPTION)
    
    print("Computing crisis flag...")
    crisis_flag_full = compute_crisis_flag(
        vix=vix_monthly,
        credit_spread=credit_spread_monthly,
        window_months=CRISIS_WINDOW_MONTHS,
        p_vix=CRISIS_VIX_PERCENTILE,
        p_cs=CRISIS_CS_PERCENTILE,
        logic=CRISIS_LOGIC,
        min_on=CRISIS_MIN_ON,
        min_off=CRISIS_MIN_OFF
    )
    
    print("Computing signals and forecasts...")
    sig_full = monthly_signal_panel(sigs_raw)
    z_full = zscore_signals(sig_full)
    betas = rolling_sectors_betas(r_full, z_full, lookback_months=LOOKBACK_BETA_MONTHS, min_months=MIN_BETA_MONTHS)
    fcast_full = sector_expected_returns(betas, z_full)

    print(" Building portfolio weights...")
    weights_full, _ = rank_and_weight_from_forecast_hysteresis(
        forecast_df=fcast_full,
        monthly_rets=r_full,
        entry_long=3,
        exit_long=6,
        dwell_min=2,
        min_bucket_size=4
    )

    weights_full = apply_position_caps_and_renormalize(
        weights_full,
        L_target=0.2,
        S_target=0.2,
        max_long_single=0.25,
        max_short_single=0.15
    )

    weights_full = apply_crisis_reduction(
        weights=weights_full,
        crisis_flag=crisis_flag_full,
        reduction=CRISIS_REDUCTION
    )

    weights_full = scale_to_target_vol(
        sector_returns=r_full,
        weights=weights_full,
        vol_target=0.12,
        lookback_months=12,
        smoothing_window=3,
        crisis_flag=crisis_flag_full
    )

    # Slice to backtest window (March 2010 - December 2024)
    weights = weights_full.loc["2010-03-01":"2024-12-31"]
    
    # Print summary
    print_sector_summary(weights)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_sector_position_counts(weights)
    plot_sector_weights_evolution(weights)
    plot_sector_activity(weights)
    
    print("\n" + "="*80)
    print("SECTOR ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
