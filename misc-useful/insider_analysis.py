"""
Insider Trading Statistical Detector!!
this analysis price + trade CSVs to detect whether trades cluster
suspiciously at local price extrema (buying at troughs, selling at peaks),
which should not be possible unless you have insider info.

"""

import csv
import os
import sys
import math
from collections import defaultdict

EXTREMA_WINDOW    = 5      # ticks on each side to qualify as local max/min
PRICE_TOLERANCE   = 0.5    # how close (in price units) a trade must be to an extremum
DAYS              = ["-2", "-1", "0"]


def load_prices(data_dir: str) -> dict[str, list[tuple[int, int, float]]]:
    """Returns {product: [(day, timestamp, mid_price), ...]} sorted by (day, ts)."""
    series: dict[str, list] = defaultdict(list)
    for d in DAYS:
        path = os.path.join(data_dir, f"prices_round_1_day_{d}.csv")
        if not os.path.exists(path):
            continue
        day_num = int(d)
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    mid = float(row['mid_price'])
                    if mid <= 0:          # skip rows with no valid mid (one-sided book)
                        continue
                    ts  = int(row['timestamp'])
                    prod = row['product'].strip()
                    series[prod].append((day_num, ts, mid))
                except (ValueError, KeyError):
                    pass
    for prod in series:
        series[prod].sort(key=lambda x: (x[0], x[1]))
    return dict(series)


def load_trades(data_dir: str) -> dict[str, list[tuple[int, int, float, int]]]:
    """Returns {product: [(day, timestamp, price, quantity), ...]} sorted by (day, ts)."""
    trades: dict[str, list] = defaultdict(list)
    for d in DAYS:
        path = os.path.join(data_dir, f"trades_round_1_day_{d}.csv")
        if not os.path.exists(path):
            continue
        day_num = int(d)
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    price = float(row['price'])
                    qty   = int(row['quantity'])
                    ts    = int(row['timestamp'])
                    prod  = row['symbol'].strip()
                    trades[prod].append((day_num, ts, price, qty))
                except (ValueError, KeyError):
                    pass
    for prod in trades:
        trades[prod].sort(key=lambda x: (x[0], x[1]))
    return dict(trades)


def find_extrema(series: list[tuple[int, int, float]], window: int) -> tuple[set, set]:
    """
    Returns (maxima_set, minima_set) where each element is (day, ts, price).
    A point is a local max if it's the highest in a ±window range (by index).
    """
    maxima = set()
    minima = set()
    prices = [p for _, _, p in series]
    n = len(prices)
    for i in range(window, n - window):
        lo = i - window
        hi = i + window + 1
        if prices[i] == max(prices[lo:hi]):
            maxima.add((series[i][0], series[i][1], prices[i]))
        if prices[i] == min(prices[lo:hi]):
            minima.add((series[i][0], series[i][1], prices[i]))
    return maxima, minima


def price_range(series: list[tuple[int, int, float]]) -> tuple[float, float]:
    prices = [p for _, _, p in series]
    return min(prices), max(prices)


def analyse_product(
    product: str,
    price_series: list[tuple[int, int, float]],
    trade_list: list[tuple[int, int, float, int]],
) -> None:
    print(f"\n{'='*60}")
    print(f"  PRODUCT: {product}")
    print(f"{'='*60}")

    if not price_series:
        print("  No price data.")
        return
    if not trade_list:
        print("  No trade data.")
        return

    pmin, pmax = price_range(price_series)
    print(f"  Price range : {pmin:.1f} – {pmax:.1f}")
    print(f"  Price ticks : {len(price_series)}")
    print(f"  Total trades: {len(trade_list)}")

    maxima, minima = find_extrema(price_series, EXTREMA_WINDOW)
    print(f"  Local maxima: {len(maxima)}  |  Local minima: {len(minima)}")

    # Build lookup: (day, ts) → price for extrema
    max_prices = {(d, ts): p for d, ts, p in maxima}
    min_prices = {(d, ts): p for d, ts, p in minima}
    all_extrema_prices = {p for _, _, p in maxima} | {p for _, _, p in minima}

    # -- Classify each trade --------------------------------------------------
    trades_at_max   = []   # trade price within tolerance of a local max
    trades_at_min   = []   # trade price within tolerance of a local min
    trades_other    = []

    for (day, ts, tprice, qty) in trade_list:
        near_max = any(abs(tprice - mp) <= PRICE_TOLERANCE for mp in
                       [p for (d, t, p) in maxima])
        near_min = any(abs(tprice - mp) <= PRICE_TOLERANCE for mp in
                       [p for (d, t, p) in minima])
        if near_max and near_min:
            trades_at_max.append((day, ts, tprice, qty))  # treat as max (ambiguous)
        elif near_max:
            trades_at_max.append((day, ts, tprice, qty))
        elif near_min:
            trades_at_min.append((day, ts, tprice, qty))
        else:
            trades_other.append((day, ts, tprice, qty))

    total = len(trade_list)
    pct_max = 100 * len(trades_at_max) / total if total else 0
    pct_min = 100 * len(trades_at_min) / total if total else 0
    pct_extrema = pct_max + pct_min

    # Expected % if trades were uniformly distributed across the price range
    # extrema_price_fraction = unique extrema prices / total unique mid-prices
    all_mid_prices = {p for _, _, p in price_series}
    frac_extrema_prices = (
        len(all_extrema_prices) / len(all_mid_prices) if all_mid_prices else 0
    )
    expected_pct = 100 * frac_extrema_prices

    print(f"\n  -- Extrema Trade Clustering ------------------------------")
    print(f"  Trades at local MAXIMA : {len(trades_at_max):4d}  ({pct_max:.1f}%)")
    print(f"  Trades at local MINIMA : {len(trades_at_min):4d}  ({pct_min:.1f}%)")
    print(f"  Trades at extrema TOTAL: {len(trades_at_max)+len(trades_at_min):4d}  ({pct_extrema:.1f}%)")
    print(f"  Random-chance baseline : {expected_pct:.1f}%  (extrema prices / all prices)")
    ratio = (pct_extrema / expected_pct) if expected_pct > 0 else float('inf')
    print(f"  Enrichment ratio       : {ratio:.2f}x  {'! SUSPICIOUS' if ratio > 2.5 else '(normal)'}")

    # -- Timing analysis: do trades happen right AT the extremum timestamp? --
    extrema_ts_set = {(d, t) for d, t, _ in maxima} | {(d, t) for d, t, _ in minima}
    trades_at_extremum_ts = [
        (day, ts, tprice, qty) for (day, ts, tprice, qty) in trade_list
        if (day, ts) in extrema_ts_set
    ]
    pct_ts_exact = 100 * len(trades_at_extremum_ts) / total if total else 0
    ts_expected  = 100 * len(extrema_ts_set) / len({(d, t) for d, t, _ in price_series})
    ts_ratio     = (pct_ts_exact / ts_expected) if ts_expected > 0 else float('inf')

    print(f"\n  -- Exact Timestamp Coincidence ---------------------------")
    print(f"  Trades at extremum timestamp: {len(trades_at_extremum_ts):4d}  ({pct_ts_exact:.1f}%)")
    print(f"  Random-chance baseline      : {ts_expected:.1f}%")
    print(f"  Enrichment ratio            : {ts_ratio:.2f}x  {'! SUSPICIOUS' if ts_ratio > 2.5 else '(normal)'}")

    # -- Day-by-day breakdown -------------------------------------------------
    print(f"\n  -- Day-by-Day Breakdown ----------------------------------")
    for day_label, day_num in [("Day -2", -2), ("Day -1", -1), ("Day  0",  0)]:
        day_trades  = [(ts, p, q) for d, ts, p, q in trade_list      if d == day_num]
        day_extrema = [(ts, p)    for d, ts, p   in (maxima | minima) if d == day_num]
        if not day_trades:
            print(f"  {day_label}: no trades")
            continue
        day_near = sum(1 for _, p, _ in day_trades
                       if any(abs(p - ep) <= PRICE_TOLERANCE for _, ep in day_extrema))
        print(f"  {day_label}: {len(day_trades):4d} trades, "
              f"{day_near:3d} near extrema "
              f"({100*day_near/len(day_trades):.1f}%),  "
              f"{len(day_extrema)} extrema ticks")

    # -- Show top suspicious trades -------------------------------------------
    suspicious = trades_at_max + trades_at_min
    if suspicious:
        print(f"\n  -- Sample Trades at Extrema (first 10) -------------------")
        print(f"  {'Day':>5}  {'Timestamp':>10}  {'Price':>10}  {'Qty':>5}  {'Type':>8}")
        shown = set()
        count = 0
        for (day, ts, tprice, qty) in sorted(suspicious, key=lambda x: (x[0], x[1])):
            if count >= 10:
                break
            typ = "MAX" if (day, ts, tprice) in [(d, t, p) for d, t, p in maxima] \
                else "MIN" if (day, ts, tprice) in [(d, t, p) for d, t, p in minima] \
                else "near-MAX" if any(abs(tprice - p) <= PRICE_TOLERANCE
                                       for _, _, p in maxima) else "near-MIN"
            print(f"  {day:>5}  {ts:>10}  {tprice:>10.1f}  {qty:>5}  {typ:>8}")
            count += 1

    # -- Verdict --------------------------------------------------------------
    print(f"\n  -- VERDICT -----------------------------------------------")
    evidence = []
    if ratio > 3.0:
        evidence.append(f"price enrichment {ratio:.1f}x above baseline")
    if ts_ratio > 3.0:
        evidence.append(f"timestamp coincidence {ts_ratio:.1f}x above baseline")
    if evidence:
        print(f"  *** POTENTIAL INSIDER SIGNAL: {'; '.join(evidence)} ***")
    elif ratio > 2.0 or ts_ratio > 2.0:
        print(f"  ! Elevated but not conclusive (price {ratio:.1f}x, timestamp {ts_ratio:.1f}x)")
    else:
        print(f"  No strong insider signal detected (price {ratio:.1f}x, timestamp {ts_ratio:.1f}x)")


# -- Main -----------------------------------------------------------------------

def main(data_dir: str) -> None:
    print(f"\nInsider Trading Detector — IMC Prosperity Round 1")
    print(f"Data directory: {data_dir}")
    print(f"Extrema window: ±{EXTREMA_WINDOW} ticks  |  Price tolerance: ±{PRICE_TOLERANCE}")

    prices = load_prices(data_dir)
    trades = load_trades(data_dir)

    products = sorted(set(prices.keys()) | set(trades.keys()))
    print(f"\nProducts found: {', '.join(products)}")

    for prod in products:
        analyse_product(
            prod,
            prices.get(prod, []),
            trades.get(prod, []),
        )

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print("  If enrichment ratio > 3x: strong suspicion of pre-knowledge.")
    print("  If enrichment ratio > 2x: worth noting, investigate further.")
    print("  Ratio ~1x: consistent with random / informed-but-not-insider trading.")
    print()


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/rucha/Downloads/ROUND1"
    main(data_dir)
