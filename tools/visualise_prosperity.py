"""
IMC Prosperity - Data Visualisation Script
==========================================
Run this script from the same folder as your data files.

Usage:
    python visualise_prosperity.py

It will look for:
  - prices_round_0_day_-1.csv   (tutorial round prices, day -1)
  - prices_round_0_day_-2.csv   (tutorial round prices, day -2)
  - trades_round_0_day_-1.csv   (market trades, day -1)
  - trades_round_0_day_-2.csv   (market trades, day -2)
  - 57971.log                   (Shriyans' submission log, optional)

All files are optional — the script plots whatever it finds.

Panels produced:
  1. Mid-price over time (both products, both days)
  2. Bid/ask spread over time
  3. Order book depth (level 1 vs level 2 volumes)
  4. Price autocorrelation / mean-reversion signal
  5. Submission PnL over time (from .log file)
  6. Trade fill scatter (price vs timestamp, coloured by side)
  7. Spread distribution histogram
  8. Rolling volatility
"""

import os, json, csv, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── Styling ──────────────────────────────────────────────────────────────────
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
    'tomato_mid':   '#58a6ff',
    'tomato_bid':   '#3fb950',
    'tomato_ask':   '#f85149',
    'emerald_mid':  '#e3b341',
    'emerald_bid':  '#a5d6a7',
    'emerald_ask':  '#ef9a9a',
    'buy_fill':     '#3fb950',
    'sell_fill':    '#f85149',
    'our_buy':      '#58a6ff',
    'our_sell':     '#ff7b72',
    'narrow':       '#ffa657',
    'pnl_tomato':   '#3fb950',
    'pnl_emerald':  '#e3b341',
    'pnl_total':    '#58a6ff',
    'signal_long':  '#238636',
    'signal_short': '#da3633',
    'vol':          '#8b949e',
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_prices(path):
    """Load a prices CSV into a DataFrame."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=';')
    for col in df.columns:
        if col not in ('product',):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_trades(path):
    """Load a trades CSV into a DataFrame."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=';')
    for col in ('price', 'quantity'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_log(path):
    """Load a Prosperity submission .log file."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        raw = json.load(f)
    lines = raw['activitiesLog'].strip().split('\n')
    reader = csv.DictReader(lines, delimiter=';')
    prices = pd.DataFrame(list(reader))
    for col in prices.columns:
        if col != 'product':
            prices[col] = pd.to_numeric(prices[col], errors='coerce')
    trades = pd.DataFrame(raw['tradeHistory'])
    if 'price' in trades.columns:
        trades['price'] = pd.to_numeric(trades['price'], errors='coerce')
        trades['quantity'] = pd.to_numeric(trades['quantity'], errors='coerce')
    return prices, trades


def combine_days(*dfs):
    """Concatenate DataFrames from multiple days, adding a global timestamp."""
    valid = [d for d in dfs if d is not None]
    if not valid:
        return None
    # Sort by day then timestamp so day -2 comes before day -1
    combined = pd.concat(valid, ignore_index=True)
    combined = combined.sort_values(['day', 'timestamp']).reset_index(drop=True)
    # Global time: offset each day so they don't overlap
    max_ts = combined['timestamp'].max()
    combined['global_ts'] = combined['day'] * (max_ts + 100) * -1 + combined['timestamp']
    combined['global_ts'] -= combined['global_ts'].min()
    return combined


# ── Signal helpers ────────────────────────────────────────────────────────────

def rolling_mean_signal(mid_series, window=20, threshold=3.0):
    """Return +1 (long), -1 (short), 0 (neutral) for each bar."""
    roll = mid_series.rolling(window, min_periods=window).mean()
    deviation = mid_series - roll
    signal = np.where(deviation < -threshold, 1,
             np.where(deviation >  threshold, -1, 0))
    return signal, roll


def rolling_vol(mid_series, window=20):
    return mid_series.diff().abs().rolling(window).mean()


# ── Main plotting ─────────────────────────────────────────────────────────────

def plot_all(prices_df, market_trades_df=None, log_prices_df=None, log_trades_df=None):

    products = ['TOMATOES', 'EMERALDS']

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle('IMC Prosperity — Tutorial Round Data Analysis', 
                 fontsize=14, fontweight='bold', color='#f0f6fc', y=0.98)

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.55, wspace=0.28,
                           left=0.06, right=0.97, top=0.95, bottom=0.04)

    # ── Panel 1: Mid-price ───────────────────────────────────────────────────
    ax_mid_t = fig.add_subplot(gs[0, 0])
    ax_mid_e = fig.add_subplot(gs[0, 1])

    for ax, product, col, label in [
        (ax_mid_t, 'TOMATOES',  'tomato_mid',  'Tomatoes'),
        (ax_mid_e, 'EMERALDS',  'emerald_mid', 'Emeralds'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        ax.plot(pdata['global_ts'], pdata['mid_price'],
                color=COLORS[col], linewidth=1.2, label='Mid price')
        ax.plot(pdata['global_ts'], pdata['bid_price_1'],
                color=COLORS[col.replace('mid','bid')], linewidth=0.6,
                alpha=0.6, linestyle='--', label='Best bid')
        ax.plot(pdata['global_ts'], pdata['ask_price_1'],
                color=COLORS[col.replace('mid','ask')], linewidth=0.6,
                alpha=0.6, linestyle='--', label='Best ask')

        # Rolling mean on tomatoes
        if product == 'TOMATOES':
            sig, roll = rolling_mean_signal(pdata['mid_price'])
            ax.plot(pdata['global_ts'], roll,
                    color='#8b949e', linewidth=0.8, linestyle=':', label='20-bar mean')
            # Shade mean-reversion regions
            for i, (ts, s) in enumerate(zip(pdata['global_ts'], sig)):
                if s == 1:
                    ax.axvspan(ts, ts + 1, alpha=0.08, color=COLORS['signal_long'])
                elif s == -1:
                    ax.axvspan(ts, ts + 1, alpha=0.08, color=COLORS['signal_short'])

        # Narrow spread events on tomatoes
        if product == 'TOMATOES':
            narrow_mask = (pdata['ask_price_1'] - pdata['bid_price_1']) < 8
            ndata = pdata[narrow_mask]
            if not ndata.empty:
                ax.scatter(ndata['global_ts'], ndata['mid_price'],
                           color=COLORS['narrow'], s=12, zorder=5,
                           label='Narrow spread (<8)', marker='D', alpha=0.7)

        # Special emerald events
        if product == 'EMERALDS':
            bid10k = pdata[pdata['bid_price_1'] == 10000]
            ask10k = pdata[pdata['ask_price_1'] == 10000]
            if not bid10k.empty:
                ax.scatter(bid10k['global_ts'], bid10k['mid_price'],
                           color='#c9a227', s=30, zorder=5,
                           label='bid=10000 (event)', marker='^')
            if not ask10k.empty:
                ax.scatter(ask10k['global_ts'], ask10k['mid_price'],
                           color='#4fc3f7', s=30, zorder=5,
                           label='ask=10000 (event)', marker='v')

        # Day boundary line
        days = pdata['day'].unique()
        if len(days) > 1:
            boundary = pdata[pdata['day'] == days[1]]['global_ts'].min()
            ax.axvline(boundary, color='#30363d', linewidth=1.5, linestyle='-')
            ax.text(boundary + 100, ax.get_ylim()[0], 'day -1 →',
                    color='#8b949e', fontsize=7, va='bottom')

        ax.set_title(f'{label} — mid-price, bid, ask')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Price')
        ax.legend(fontsize=7, loc='best', ncol=2)

    # ── Panel 2: Spread over time ────────────────────────────────────────────
    ax_sp_t = fig.add_subplot(gs[1, 0])
    ax_sp_e = fig.add_subplot(gs[1, 1])

    for ax, product, col, label in [
        (ax_sp_t, 'TOMATOES', COLORS['tomato_mid'],  'Tomatoes'),
        (ax_sp_e, 'EMERALDS', COLORS['emerald_mid'], 'Emeralds'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        spread = pdata['ask_price_1'] - pdata['bid_price_1']
        ax.fill_between(pdata['global_ts'], spread, alpha=0.4, color=col)
        ax.plot(pdata['global_ts'], spread, color=col, linewidth=0.8)
        ax.axhline(spread.mean(), color='#8b949e', linewidth=1,
                   linestyle='--', label=f'Mean {spread.mean():.1f}')
        if product == 'TOMATOES':
            ax.axhline(8, color=COLORS['narrow'], linewidth=1,
                       linestyle=':', alpha=0.8, label='Narrow threshold (8)')
        ax.set_title(f'{label} — bid/ask spread (ticks)')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Spread (ticks)')
        ax.legend(fontsize=7)

    # ── Panel 3: Order book depth ────────────────────────────────────────────
    ax_depth_t = fig.add_subplot(gs[2, 0])
    ax_depth_e = fig.add_subplot(gs[2, 1])

    for ax, product, col, label in [
        (ax_depth_t, 'TOMATOES', COLORS['tomato_mid'],  'Tomatoes'),
        (ax_depth_e, 'EMERALDS', COLORS['emerald_mid'], 'Emeralds'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        ax.fill_between(pdata['global_ts'], pdata['bid_volume_1'].fillna(0),
                        alpha=0.5, color=COLORS['tomato_bid'], label='Bid vol L1')
        ax.fill_between(pdata['global_ts'], pdata['bid_volume_2'].fillna(0),
                        alpha=0.25, color=COLORS['tomato_bid'], label='Bid vol L2')
        ax.fill_between(pdata['global_ts'], -pdata['ask_volume_1'].fillna(0),
                        alpha=0.5, color=COLORS['tomato_ask'], label='Ask vol L1')
        ax.fill_between(pdata['global_ts'], -pdata['ask_volume_2'].fillna(0),
                        alpha=0.25, color=COLORS['tomato_ask'], label='Ask vol L2')
        ax.axhline(0, color='#30363d', linewidth=0.8)
        ax.set_title(f'{label} — order book depth (buy +, sell −)')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Volume')
        ax.legend(fontsize=7, ncol=2)

    # ── Panel 4: Mean-reversion signal (tomatoes only) ───────────────────────
    ax_mr = fig.add_subplot(gs[3, 0])
    pdata = prices_df[prices_df['product'] == 'TOMATOES'].copy()
    if not pdata.empty:
        sig, roll = rolling_mean_signal(pdata['mid_price'], window=20, threshold=3.0)
        deviation = pdata['mid_price'] - roll

        ax_mr.fill_between(pdata['global_ts'], deviation,
                           where=(deviation < 0), alpha=0.5,
                           color=COLORS['signal_long'], label='Below mean (lean long)')
        ax_mr.fill_between(pdata['global_ts'], deviation,
                           where=(deviation >= 0), alpha=0.5,
                           color=COLORS['signal_short'], label='Above mean (lean short)')
        ax_mr.axhline(3,  color=COLORS['signal_short'], linewidth=1, linestyle='--', alpha=0.6)
        ax_mr.axhline(-3, color=COLORS['signal_long'],  linewidth=1, linestyle='--', alpha=0.6)
        ax_mr.axhline(0,  color='#30363d', linewidth=1)
        ax_mr.set_title('Tomatoes — deviation from 20-bar rolling mean (MR signal)')
        ax_mr.set_xlabel('Timestamp')
        ax_mr.set_ylabel('Deviation (ticks)')
        ax_mr.legend(fontsize=7)

    # ── Panel 5: Rolling volatility ──────────────────────────────────────────
    ax_vol = fig.add_subplot(gs[3, 1])
    for product, col, label in [
        ('TOMATOES', COLORS['tomato_mid'],  'Tomatoes'),
        ('EMERALDS', COLORS['emerald_mid'], 'Emeralds'),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if not pdata.empty:
            vol = rolling_vol(pdata['mid_price'], window=20)
            ax_vol.plot(pdata['global_ts'], vol, color=col,
                        linewidth=1, label=label, alpha=0.8)
    ax_vol.set_title('Rolling 20-bar mid-price volatility (|Δmid|)')
    ax_vol.set_xlabel('Timestamp')
    ax_vol.set_ylabel('Avg |Δprice|')
    ax_vol.legend(fontsize=7)

    # ── Panel 6: PnL (from log if available, else market trades estimate) ────
    ax_pnl = fig.add_subplot(gs[4, 0])
    if log_prices_df is not None:
        for product, col, label in [
            ('TOMATOES', COLORS['pnl_tomato'],  'Tomatoes PnL'),
            ('EMERALDS', COLORS['pnl_emerald'], 'Emeralds PnL'),
        ]:
            ldata = log_prices_df[log_prices_df['product'] == product].copy()
            ldata = ldata.sort_values('timestamp')
            if not ldata.empty:
                combined_pnl = ldata['profit_and_loss'].fillna(0)
                ax_pnl.plot(ldata['timestamp'], combined_pnl,
                            color=col, linewidth=1.2, label=label)
        # Total
        tpnl = log_prices_df[log_prices_df['product']=='TOMATOES'].set_index('timestamp')['profit_and_loss']
        epnl = log_prices_df[log_prices_df['product']=='EMERALDS'].set_index('timestamp')['profit_and_loss']
        combined = (tpnl + epnl).dropna()
        ax_pnl.plot(combined.index, combined.values,
                    color=COLORS['pnl_total'], linewidth=1.5,
                    linestyle='--', label='Total PnL')
        ax_pnl.set_title('Submission PnL over time (from .log)')
    else:
        ax_pnl.text(0.5, 0.5, 'No .log file found\n(place 57971.log here)',
                    ha='center', va='center', color='#8b949e', fontsize=9,
                    transform=ax_pnl.transAxes)
        ax_pnl.set_title('Submission PnL (log not found)')
    ax_pnl.set_xlabel('Timestamp')
    ax_pnl.set_ylabel('PnL')
    ax_pnl.legend(fontsize=7)

    # ── Panel 7: Trade fills scatter ─────────────────────────────────────────
    ax_fills = fig.add_subplot(gs[4, 1])
    plotted_any = False

    # Market trades (from CSV files)
    if market_trades_df is not None:
        for symbol, col in [('TOMATOES', COLORS['tomato_mid']),
                             ('EMERALDS', COLORS['emerald_mid'])]:
            sdata = market_trades_df[market_trades_df['symbol'] == symbol]
            if not sdata.empty:
                ax_fills.scatter(sdata['timestamp'], sdata['price'],
                                 color=col, s=8, alpha=0.5, label=f'{symbol} market')
                plotted_any = True

    # Our fills (from log)
    if log_trades_df is not None and 'buyer' in log_trades_df.columns:
        our_buys  = log_trades_df[log_trades_df['buyer']  == 'SUBMISSION']
        our_sells = log_trades_df[log_trades_df['seller'] == 'SUBMISSION']
        if not our_buys.empty:
            ax_fills.scatter(our_buys['timestamp'], our_buys['price'],
                             color=COLORS['our_buy'], s=50, zorder=6,
                             marker='^', label='Our buys', edgecolors='white', linewidths=0.5)
            plotted_any = True
        if not our_sells.empty:
            ax_fills.scatter(our_sells['timestamp'], our_sells['price'],
                             color=COLORS['our_sell'], s=50, zorder=6,
                             marker='v', label='Our sells', edgecolors='white', linewidths=0.5)
            plotted_any = True

    if not plotted_any:
        ax_fills.text(0.5, 0.5, 'No trade data found',
                      ha='center', va='center', color='#8b949e',
                      transform=ax_fills.transAxes)
    ax_fills.set_title('Trade fills (market + our submissions)')
    ax_fills.set_xlabel('Timestamp')
    ax_fills.set_ylabel('Fill price')
    ax_fills.legend(fontsize=7, ncol=2)

    plt.savefig('prosperity_analysis.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Saved: prosperity_analysis.png")
    plt.show()


# ── Spread distribution (separate figure) ────────────────────────────────────

def plot_distributions(prices_df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('IMC Prosperity — Statistical Distributions', 
                 fontsize=13, fontweight='bold', color='#f0f6fc')

    for axes_row, product, col in [
        (axes[0], 'TOMATOES', COLORS['tomato_mid']),
        (axes[1], 'EMERALDS', COLORS['emerald_mid']),
    ]:
        pdata = prices_df[prices_df['product'] == product].copy()
        if pdata.empty:
            continue
        spread = pdata['ask_price_1'] - pdata['bid_price_1']
        diffs  = pdata['mid_price'].diff().dropna()

        # Spread histogram
        ax = axes_row[0]
        ax.set_facecolor('#161b22')
        counts, bins, patches = ax.hist(spread.dropna(), bins=25,
                                         color=col, alpha=0.75, edgecolor='#30363d')
        ax.axvline(spread.mean(), color='white', linewidth=1.5, linestyle='--',
                   label=f'Mean={spread.mean():.1f}')
        if product == 'TOMATOES':
            ax.axvline(8, color=COLORS['narrow'], linewidth=1.5,
                       linestyle=':', label='Narrow threshold (8)')
        ax.set_title(f'{product} — spread distribution')
        ax.set_xlabel('Spread (ticks)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

        # Price diff histogram (1-bar returns)
        ax = axes_row[1]
        ax.set_facecolor('#161b22')
        ax.hist(diffs, bins=40, color=col, alpha=0.75, edgecolor='#30363d')
        ax.axvline(0, color='white', linewidth=1)
        ax.axvline(diffs.mean(), color='#e3b341', linewidth=1.5, linestyle='--',
                   label=f'Mean={diffs.mean():.3f}')
        ax.set_title(f'{product} — 1-bar price changes')
        ax.set_xlabel('Δ mid-price')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('prosperity_distributions.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Saved: prosperity_distributions.png")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("IMC Prosperity — Data Visualiser")
    print("=" * 40)

    # Load price data (CSV files from tutorial round download)
    p1 = load_prices('prices_round_0_day_-1.csv')
    p2 = load_prices('prices_round_0_day_-2.csv')

    if p1 is None and p2 is None:
        print("ERROR: No price CSV files found.")
        print("Expected: prices_round_0_day_-1.csv and/or prices_round_0_day_-2.csv")
        return

    prices_df = combine_days(p2, p1)
    print(f"Loaded {len(prices_df)} price rows across {prices_df['day'].nunique()} day(s)")

    # Load market trade data
    t1 = load_trades('trades_round_0_day_-1.csv')
    t2 = load_trades('trades_round_0_day_-2.csv')
    market_trades = pd.concat([df for df in [t2, t1] if df is not None], ignore_index=True) \
                    if any(x is not None for x in [t1, t2]) else None
    if market_trades is not None:
        print(f"Loaded {len(market_trades)} market trade rows")

    # Load submission log (optional)
    log_prices, log_trades = load_log('57971.log')
    if log_prices is not None:
        print(f"Loaded submission log: {len(log_prices)} price rows, {len(log_trades)} trades")
    else:
        print("No submission log found (57971.log) — PnL panel will be empty")

    print()
    print("Generating plots...")
    plot_all(prices_df, market_trades, log_prices, log_trades)
    plot_distributions(prices_df)
    print("\nDone! Check prosperity_analysis.png and prosperity_distributions.png")


if __name__ == '__main__':
    main()
