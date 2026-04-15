"""
IMC Prosperity Round 1 — Data Visualisation
=============================================
Products: ASH_COATED_OSMIUM (volatile, hidden pattern)
          INTARIAN_PEPPER_ROOT (stable, like Emeralds)

Usage:
    python tools/visualise_round1.py

Reads from: C:/Users/rucha/Downloads/ROUND1/
Saves:      tools/round1_analysis.png
            tools/round1_distributions.png
            tools/round1_osmium_pattern.png
"""

import os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "C:/Users/rucha/Downloads/ROUND1"
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'axes.titlecolor':  '#f0f6fc',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'axes.grid':        True,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family':      'monospace',
    'axes.titlesize':   11,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
})

COLORS = {
    'osmium_mid':   '#58a6ff',
    'osmium_bid':   '#3fb950',
    'osmium_ask':   '#f85149',
    'pepper_mid':   '#e3b341',
    'pepper_bid':   '#a5d6a7',
    'pepper_ask':   '#ef9a9a',
    'signal_long':  '#238636',
    'signal_short': '#da3633',
    'vol':          '#8b949e',
    'fft':          '#ff7b72',
    'sine_fit':     '#ffa657',
}

OSMIUM  = 'ASH_COATED_OSMIUM'
PEPPER  = 'INTARIAN_PEPPER_ROOT'
DAYS    = [-2, -1, 0]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices(day):
    path = os.path.join(DATA_DIR, f"prices_round_1_day_{day}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=';')
    for col in df.columns:
        if col != 'product':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_trades(day):
    path = os.path.join(DATA_DIR, f"trades_round_1_day_{day}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=';')
    for col in ('price', 'quantity'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def combine_days(*dfs):
    valid = [d for d in dfs if d is not None]
    if not valid:
        return None
    combined = pd.concat(valid, ignore_index=True)
    # Filter out rows where mid_price is 0 (one-sided book artefact)
    combined = combined[combined['mid_price'] > 100].copy()
    combined = combined.sort_values(['day', 'timestamp']).reset_index(drop=True)
    max_ts = 1_000_000
    combined['global_ts'] = (combined['day'] + 2) * (max_ts + 100) + combined['timestamp']
    combined['global_ts'] -= combined['global_ts'].min()
    return combined


# ── Signal helpers ────────────────────────────────────────────────────────────

def rolling_vol(mid_series, window=50):
    return mid_series.diff().abs().rolling(window).mean()


def find_dominant_period(mid_series):
    """Return dominant period (in samples) via FFT, skipping DC."""
    x = mid_series.dropna().values
    x = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    # Skip DC (index 0)
    fft_vals[0] = 0
    dominant_freq = freqs[np.argmax(fft_vals)]
    period = 1.0 / dominant_freq if dominant_freq > 0 else None
    return period, freqs, fft_vals


def fit_sine(ts, mid_series, period_samples):
    """Fit A*sin(2π/T * t + φ) + offset to the series."""
    x = mid_series.dropna().values
    t = np.arange(len(x))
    omega = 2 * math.pi / period_samples

    # Linearise: y = A*sin(ωt) + B*cos(ωt) + C
    S = np.sin(omega * t)
    C_ = np.cos(omega * t)
    A_mat = np.column_stack([S, C_, np.ones(len(x))])
    coeffs, *_ = np.linalg.lstsq(A_mat, x, rcond=None)
    A_sin, B_cos, offset = coeffs
    amplitude = math.sqrt(A_sin**2 + B_cos**2)
    phase = math.atan2(B_cos, A_sin)
    fitted = amplitude * np.sin(omega * t + phase) + offset
    residuals = x - fitted
    return fitted, residuals, amplitude, offset, phase


# ── Plot 1: Main overview ─────────────────────────────────────────────────────

def plot_overview(prices_df, market_trades_df=None):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('IMC Prosperity Round 1 — Market Data Analysis',
                 fontsize=15, fontweight='bold', color='#f0f6fc', y=0.99)
    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.55, wspace=0.28,
                           left=0.06, right=0.97, top=0.96, bottom=0.03)

    day_boundaries = []
    for i, d in enumerate(DAYS[1:], 1):
        mask = prices_df['day'] == d
        if mask.any():
            day_boundaries.append((prices_df[mask]['global_ts'].min(), f'Day {d}'))

    def add_day_lines(ax):
        for ts, label in day_boundaries:
            ax.axvline(ts, color='#30363d', linewidth=1.2, linestyle='-')
            ylim = ax.get_ylim()
            ax.text(ts + 2000, ylim[0] + (ylim[1]-ylim[0])*0.03,
                    label, color='#8b949e', fontsize=7, va='bottom')

    # ── Panel 0: Mid-price both products ────────────────────────────────────
    ax_osm = fig.add_subplot(gs[0, 0])
    ax_pep = fig.add_subplot(gs[0, 1])

    for ax, product, ckey, label in [
        (ax_osm, OSMIUM, 'osmium', 'ASH_COATED_OSMIUM'),
        (ax_pep, PEPPER, 'pepper', 'INTARIAN_PEPPER_ROOT'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        ax.plot(pdata['global_ts'], pdata['mid_price'],
                color=COLORS[f'{ckey}_mid'], linewidth=1.0, label='Mid price')
        ax.plot(pdata['global_ts'], pdata['bid_price_1'],
                color=COLORS[f'{ckey}_bid'], linewidth=0.5,
                alpha=0.5, linestyle='--', label='Best bid')
        ax.plot(pdata['global_ts'], pdata['ask_price_1'],
                color=COLORS[f'{ckey}_ask'], linewidth=0.5,
                alpha=0.5, linestyle='--', label='Best ask')

        # Rolling mean overlay
        roll = pdata['mid_price'].rolling(100, min_periods=20).mean()
        ax.plot(pdata['global_ts'], roll,
                color='#8b949e', linewidth=1.0, linestyle=':', alpha=0.9,
                label='100-bar rolling mean')

        add_day_lines(ax)
        ax.set_title(f'{label} — mid-price')
        ax.set_xlabel('Global timestamp')
        ax.set_ylabel('Price')
        ax.legend(fontsize=7, loc='best', ncol=2)

    # ── Panel 1: Bid/ask spread ──────────────────────────────────────────────
    ax_sp_o = fig.add_subplot(gs[1, 0])
    ax_sp_p = fig.add_subplot(gs[1, 1])

    for ax, product, ckey, label in [
        (ax_sp_o, OSMIUM, 'osmium', 'ASH_COATED_OSMIUM'),
        (ax_sp_p, PEPPER, 'pepper', 'INTARIAN_PEPPER_ROOT'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        spread = pdata['ask_price_1'] - pdata['bid_price_1']
        ax.fill_between(pdata['global_ts'], spread.fillna(0),
                        alpha=0.35, color=COLORS[f'{ckey}_mid'])
        ax.plot(pdata['global_ts'], spread, color=COLORS[f'{ckey}_mid'], linewidth=0.7)
        ax.axhline(spread.mean(), color='#8b949e', linewidth=1, linestyle='--',
                   label=f'Mean {spread.mean():.1f}')
        add_day_lines(ax)
        ax.set_title(f'{label} — bid/ask spread')
        ax.set_xlabel('Global timestamp')
        ax.set_ylabel('Spread (ticks)')
        ax.legend(fontsize=7)

    # ── Panel 2: Order book depth ────────────────────────────────────────────
    ax_dep_o = fig.add_subplot(gs[2, 0])
    ax_dep_p = fig.add_subplot(gs[2, 1])

    for ax, product, ckey, label in [
        (ax_dep_o, OSMIUM, 'osmium', 'ASH_COATED_OSMIUM'),
        (ax_dep_p, PEPPER, 'pepper', 'INTARIAN_PEPPER_ROOT'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        ax.fill_between(pdata['global_ts'], pdata['bid_volume_1'].fillna(0),
                        alpha=0.55, color=COLORS[f'{ckey}_bid'], label='Bid vol L1')
        ax.fill_between(pdata['global_ts'], pdata['bid_volume_2'].fillna(0),
                        alpha=0.25, color=COLORS[f'{ckey}_bid'], label='Bid vol L2')
        ax.fill_between(pdata['global_ts'], -pdata['ask_volume_1'].fillna(0),
                        alpha=0.55, color=COLORS[f'{ckey}_ask'], label='Ask vol L1')
        ax.fill_between(pdata['global_ts'], -pdata['ask_volume_2'].fillna(0),
                        alpha=0.25, color=COLORS[f'{ckey}_ask'], label='Ask vol L2')
        ax.axhline(0, color='#30363d', linewidth=0.8)
        add_day_lines(ax)
        ax.set_title(f'{label} — order book depth')
        ax.set_xlabel('Global timestamp')
        ax.set_ylabel('Volume (buy +, sell −)')
        ax.legend(fontsize=7, ncol=2)

    # ── Panel 3: Mean-reversion deviation ────────────────────────────────────
    ax_mr_o = fig.add_subplot(gs[3, 0])
    ax_mr_p = fig.add_subplot(gs[3, 1])

    for ax, product, ckey, label, window in [
        (ax_mr_o, OSMIUM, 'osmium', 'ASH_COATED_OSMIUM',  200),
        (ax_mr_p, PEPPER, 'pepper', 'INTARIAN_PEPPER_ROOT', 50),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        roll = pdata['mid_price'].rolling(window, min_periods=window//2).mean()
        dev  = pdata['mid_price'] - roll
        thresh = dev.std() * 1.5

        ax.fill_between(pdata['global_ts'], dev,
                        where=(dev < 0), alpha=0.45,
                        color=COLORS['signal_long'], label='Below mean (lean long)')
        ax.fill_between(pdata['global_ts'], dev,
                        where=(dev >= 0), alpha=0.45,
                        color=COLORS['signal_short'], label='Above mean (lean short)')
        ax.axhline( thresh, color=COLORS['signal_short'],
                    linewidth=1, linestyle='--', alpha=0.7, label=f'+{thresh:.1f} σ threshold')
        ax.axhline(-thresh, color=COLORS['signal_long'],
                   linewidth=1, linestyle='--', alpha=0.7, label=f'-{thresh:.1f} σ threshold')
        ax.axhline(0, color='#30363d', linewidth=1)
        add_day_lines(ax)
        ax.set_title(f'{label} — deviation from {window}-bar mean')
        ax.set_xlabel('Global timestamp')
        ax.set_ylabel('Deviation (ticks)')
        ax.legend(fontsize=7, ncol=2)

    # ── Panel 4: Rolling volatility comparison ────────────────────────────────
    ax_vol = fig.add_subplot(gs[4, 0])
    for product, ckey, label in [
        (OSMIUM, 'osmium', 'ASH_COATED_OSMIUM'),
        (PEPPER, 'pepper', 'INTARIAN_PEPPER_ROOT'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if not pdata.empty:
            vol = rolling_vol(pdata['mid_price'], window=50)
            ax_vol.plot(pdata['global_ts'], vol,
                        color=COLORS[f'{ckey}_mid'], linewidth=1.0,
                        label=label.split('_')[0], alpha=0.9)
    add_day_lines(ax_vol)
    ax_vol.set_title('Rolling 50-bar volatility (|Δmid|) — both products')
    ax_vol.set_xlabel('Global timestamp')
    ax_vol.set_ylabel('Avg |Δprice|')
    ax_vol.legend(fontsize=7)

    # ── Panel 5: Market trade scatter ────────────────────────────────────────
    ax_trades = fig.add_subplot(gs[4, 1])
    if market_trades_df is not None:
        for product, ckey in [(OSMIUM, 'osmium'), (PEPPER, 'pepper')]:
            td = market_trades_df[market_trades_df['symbol'] == product]
            if not td.empty:
                # global_ts for trades: combine from day col if present
                if 'day' in td.columns:
                    td = td.copy()
                    td['global_ts'] = (td['day'] + 2) * 1_000_100 + td['timestamp']
                    td['global_ts'] -= td['global_ts'].min()
                    ax_trades.scatter(td['global_ts'], td['price'],
                                      color=COLORS[f'{ckey}_mid'], s=6,
                                      alpha=0.4, label=product.split('_')[0])
                else:
                    ax_trades.scatter(td['timestamp'], td['price'],
                                      color=COLORS[f'{ckey}_mid'], s=6,
                                      alpha=0.4, label=product.split('_')[0])
    else:
        ax_trades.text(0.5, 0.5, 'No trade data', ha='center', va='center',
                       color='#8b949e', transform=ax_trades.transAxes)
    ax_trades.set_title('Market trade fills (price vs time)')
    ax_trades.set_xlabel('Timestamp')
    ax_trades.set_ylabel('Price')
    ax_trades.legend(fontsize=7)

    out = os.path.join(OUT_DIR, 'round1_analysis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


# ── Plot 2: Distributions ─────────────────────────────────────────────────────

def plot_distributions(prices_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Round 1 — Statistical Distributions',
                 fontsize=13, fontweight='bold', color='#f0f6fc')

    for axes_row, product, ckey in [
        (axes[0], OSMIUM, 'osmium'),
        (axes[1], PEPPER, 'pepper'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        spread = pdata['ask_price_1'] - pdata['bid_price_1']
        diffs  = pdata['mid_price'].diff().dropna()

        ax = axes_row[0]
        ax.set_facecolor('#161b22')
        ax.hist(spread.dropna(), bins=30, color=COLORS[f'{ckey}_mid'],
                alpha=0.75, edgecolor='#30363d')
        ax.axvline(spread.mean(), color='white', linewidth=1.5, linestyle='--',
                   label=f'Mean={spread.mean():.1f}')
        ax.set_title(f'{product} — spread distribution')
        ax.set_xlabel('Spread (ticks)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

        ax = axes_row[1]
        ax.set_facecolor('#161b22')
        ax.hist(diffs, bins=50, color=COLORS[f'{ckey}_mid'],
                alpha=0.75, edgecolor='#30363d')
        ax.axvline(0, color='white', linewidth=1)
        ax.axvline(diffs.mean(), color='#e3b341', linewidth=1.5, linestyle='--',
                   label=f'Mean={diffs.mean():.3f}')
        ax.set_title(f'{product} — 1-bar price changes')
        ax.set_xlabel('Δ mid-price')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, 'round1_distributions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


# ── Plot 3: Osmium hidden-pattern deep-dive ────────────────────────────────

def plot_osmium_pattern(prices_df):
    pdata = prices_df[prices_df['product'] == OSMIUM].copy().reset_index(drop=True)
    if pdata.empty:
        print("No ASH_COATED_OSMIUM data for pattern analysis.")
        return

    mid = pdata['mid_price'].interpolate()

    # FFT
    period_samples, freqs, fft_vals = find_dominant_period(mid)
    print(f"\n[ASH_COATED_OSMIUM] Dominant period: {period_samples:.1f} samples")

    # Sine fit on full series
    fitted, residuals, amp, offset, phase = fit_sine(pdata['global_ts'], mid, period_samples)
    print(f"[ASH_COATED_OSMIUM] Sine fit: amplitude={amp:.2f}, offset={offset:.2f}")
    print(f"[ASH_COATED_OSMIUM] Residual std: {residuals.std():.4f}")

    # Autocorrelation
    max_lag = 500
    lags = range(1, min(max_lag, len(mid) // 3))
    acf = [mid.autocorr(lag=l) for l in lags]

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('ASH_COATED_OSMIUM — Hidden Pattern Analysis',
                 fontsize=14, fontweight='bold', color='#f0f6fc', y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.28,
                           left=0.06, right=0.97, top=0.96, bottom=0.03)

    # Panel A: raw mid + fitted sine
    ax = fig.add_subplot(gs[0, :])
    ax.plot(pdata['global_ts'], mid.values,
            color=COLORS['osmium_mid'], linewidth=0.8, label='Mid price', alpha=0.8)
    ax.plot(pdata['global_ts'][:len(fitted)], fitted,
            color=COLORS['sine_fit'], linewidth=1.5, linestyle='--',
            label=f'Best-fit sine  (A={amp:.1f}, T={period_samples:.0f} bars, offset={offset:.0f})')
    ax.set_title('Mid-price + fitted sinusoidal model (all days)')
    ax.set_xlabel('Global timestamp')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8)

    # Panel B: residuals after sine removal
    ax = fig.add_subplot(gs[1, 0])
    t_vals = pdata['global_ts'][:len(residuals)]
    ax.fill_between(t_vals, residuals,
                    where=(np.array(residuals) < 0), alpha=0.5,
                    color=COLORS['signal_long'], label='Below fit')
    ax.fill_between(t_vals, residuals,
                    where=(np.array(residuals) >= 0), alpha=0.5,
                    color=COLORS['signal_short'], label='Above fit')
    ax.axhline(0, color='#30363d', linewidth=1)
    ax.set_title(f'Residuals after sine removal (std={residuals.std():.2f})')
    ax.set_xlabel('Global timestamp')
    ax.set_ylabel('Residual (ticks)')
    ax.legend(fontsize=7)

    # Panel C: FFT power spectrum
    ax = fig.add_subplot(gs[1, 1])
    # Only plot positive freqs above ~0.0001 to avoid DC
    mask = freqs > 0.0002
    ax.plot(1.0 / freqs[mask], fft_vals[mask],
            color=COLORS['fft'], linewidth=1.0)
    ax.axvline(period_samples, color='white', linewidth=1.5, linestyle='--',
               label=f'Dominant period = {period_samples:.0f} bars')
    ax.set_title('FFT power spectrum (period in samples)')
    ax.set_xlabel('Period (bars)')
    ax.set_ylabel('FFT magnitude')
    ax.set_xlim(0, min(period_samples * 5, len(mid) // 2))
    ax.legend(fontsize=8)

    # Panel D: autocorrelation
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(list(lags), acf, color=COLORS['osmium_mid'], linewidth=0.8)
    ax.axhline(0, color='#30363d', linewidth=1)
    ax.axhline( 0.05, color='#8b949e', linewidth=0.8, linestyle=':')
    ax.axhline(-0.05, color='#8b949e', linewidth=0.8, linestyle=':')
    # Mark first strong positive peak
    acf_arr = np.array(acf)
    if acf_arr.max() > 0.1:
        peak_lag = list(lags)[np.argmax(acf_arr)]
        ax.axvline(peak_lag, color=COLORS['sine_fit'], linewidth=1.5,
                   linestyle='--', label=f'Peak ACF @ lag {peak_lag}')
    ax.set_title('Autocorrelation function (up to lag 500)')
    ax.set_xlabel('Lag (bars)')
    ax.set_ylabel('ACF')
    ax.legend(fontsize=7)

    # Panel E: per-day mid-price with sine overlay
    ax = fig.add_subplot(gs[2, 1])
    day_colors = ['#58a6ff', '#e3b341', '#3fb950']
    for i, day in enumerate(DAYS):
        ddata = pdata[pdata['day'] == day]
        if ddata.empty:
            continue
        ax.plot(ddata['timestamp'], ddata['mid_price'],
                color=day_colors[i], linewidth=0.8, label=f'Day {day}', alpha=0.9)
    ax.set_title('ASH_COATED_OSMIUM — mid-price per day (aligned)')
    ax.set_xlabel('Timestamp (within day)')
    ax.set_ylabel('Price')
    ax.legend(fontsize=7)

    # Panel F: rolling 100-bar z-score (mean-reversion signal)
    ax = fig.add_subplot(gs[3, 0])
    roll_mean = mid.rolling(200, min_periods=50).mean()
    roll_std  = mid.rolling(200, min_periods=50).std()
    zscore = (mid - roll_mean) / roll_std.replace(0, np.nan)
    ax.plot(pdata['global_ts'], zscore, color=COLORS['osmium_mid'], linewidth=0.8)
    ax.axhline( 2, color=COLORS['signal_short'], linewidth=1, linestyle='--', label='+2σ')
    ax.axhline(-2, color=COLORS['signal_long'],  linewidth=1, linestyle='--', label='-2σ')
    ax.axhline(0, color='#30363d', linewidth=1)
    ax.set_title('200-bar rolling z-score (mean-reversion signal)')
    ax.set_xlabel('Global timestamp')
    ax.set_ylabel('Z-score')
    ax.legend(fontsize=7)

    # Panel G: residual histogram
    ax = fig.add_subplot(gs[3, 1])
    ax.set_facecolor('#161b22')
    ax.hist(residuals, bins=50, color=COLORS['fft'], alpha=0.75, edgecolor='#30363d')
    ax.axvline(0, color='white', linewidth=1)
    ax.set_title(f'Residual distribution (std={residuals.std():.2f})')
    ax.set_xlabel('Residual value')
    ax.set_ylabel('Count')

    out = os.path.join(OUT_DIR, 'round1_osmium_pattern.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)

    return period_samples, amp, offset, residuals.std()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("IMC Prosperity Round 1 — Data Visualiser")
    print("=" * 45)

    price_frames = [load_prices(d) for d in DAYS]
    loaded_days  = [d for d, f in zip(DAYS, price_frames) if f is not None]
    prices_df    = combine_days(*[f for f in price_frames if f is not None])

    if prices_df is None:
        print(f"ERROR: No price CSV files found in {DATA_DIR}")
        return
    print(f"Loaded {len(prices_df)} price rows across days: {loaded_days}")
    print(f"Products: {prices_df['product'].unique().tolist()}")

    trade_frames = [load_trades(d) for d in DAYS]
    market_trades = None
    valid_trades  = [f for f in trade_frames if f is not None]
    if valid_trades:
        market_trades = pd.concat(valid_trades, ignore_index=True)
        print(f"Loaded {len(market_trades)} market trade rows")

    print("\nGenerating overview plots...")
    plot_overview(prices_df, market_trades)
    plot_distributions(prices_df)

    print("\nRunning OSMium pattern analysis...")
    result = plot_osmium_pattern(prices_df)
    if result:
        period, amp, offset, res_std = result
        print(f"\n{'='*45}")
        print(f"OSMIUM PATTERN SUMMARY")
        print(f"  Dominant period : {period:.0f} bars (~{period*100/1000:.1f}s at 100ms/bar)")
        print(f"  Amplitude       : {amp:.2f} ticks")
        print(f"  Mid-price offset: {offset:.2f}")
        print(f"  Residual std    : {res_std:.4f} ticks")

    # Print basic stats
    print(f"\n{'='*45}")
    print("BASIC PRICE STATISTICS")
    for product in [OSMIUM, PEPPER]:
        pdata = prices_df[prices_df['product'] == product]
        if pdata.empty:
            continue
        mid = pdata['mid_price']
        spread = pdata['ask_price_1'] - pdata['bid_price_1']
        print(f"\n{product}:")
        print(f"  Mid price  — mean={mid.mean():.2f}, std={mid.std():.4f}, "
              f"min={mid.min():.1f}, max={mid.max():.1f}, range={mid.max()-mid.min():.1f}")
        print(f"  Spread     — mean={spread.mean():.2f}, std={spread.std():.2f}, "
              f"min={spread.min():.1f}, max={spread.max():.1f}")

    print("\nDone! Check tools/round1_*.png")


if __name__ == '__main__':
    main()
