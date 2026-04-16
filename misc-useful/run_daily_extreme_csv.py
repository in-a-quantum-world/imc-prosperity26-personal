"""
Replay DailyExtremeTracker on Round 1 CSV data.
Usage: python run_daily_extreme_csv.py [path_to_data_dir]
"""

import csv, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from daily_extreme_tracker import (
    DailyExtremeTracker, ProductExtremeState,
    LONG, SHORT, NEUTRAL, PRICE_TOLERANCE
)

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/rucha/Downloads/ROUND1"
DAYS     = [("-2", -2), ("-1", -1), ("0", 0)]
CONFIRM_TICKS = 10   # ticks ahead to check if signal was correct

# ── Mock classes ──────────────────────────────────────────────────────────────

class MockTrade:
    def __init__(self, price, quantity):
        self.price    = price
        self.quantity = quantity

class MockOrderDepth:
    def __init__(self, bids, asks):
        self.buy_orders  = bids
        self.sell_orders = asks

class MockState:
    def __init__(self, ts, depths, market_trades):
        self.timestamp     = ts
        self.order_depths  = depths
        self.market_trades = market_trades
        self.own_trades    = {}
        self.position      = {}

# ── Load data ─────────────────────────────────────────────────────────────────

def load_price_ticks(data_dir):
    """Returns {(day_num, ts, product): (MockOrderDepth, mid)}"""
    ticks = {}
    for (d_str, d_num) in DAYS:
        path = f"{data_dir}/prices_round_1_day_{d_str}.csv"
        if not os.path.exists(path): continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                prod = row['product'].strip()
                try:
                    mid = float(row['mid_price'])
                    if mid <= 0: continue
                    ts  = int(row['timestamp'])
                    bids, asks = {}, {}
                    for i in range(1, 4):
                        bp = row.get(f'bid_price_{i}','').strip()
                        bv = row.get(f'bid_volume_{i}','').strip()
                        if bp and bv: bids[float(bp)] = int(bv)
                        ap = row.get(f'ask_price_{i}','').strip()
                        av = row.get(f'ask_volume_{i}','').strip()
                        if ap and av: asks[float(ap)] = -int(av)
                    if not bids or not asks: continue
                    ticks[(d_num, ts, prod)] = (MockOrderDepth(bids, asks), mid)
                except: pass
    return ticks

def load_trade_ticks(data_dir):
    """Returns {(day_num, ts, product): [MockTrade, ...]}"""
    trades = defaultdict(list)
    for (d_str, d_num) in DAYS:
        path = f"{data_dir}/trades_round_1_day_{d_str}.csv"
        if not os.path.exists(path): continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                try:
                    ts    = int(row['timestamp'])
                    price = float(row['price'])
                    qty   = int(row['quantity'])
                    prod  = row['symbol'].strip()
                    trades[(d_num, ts, prod)].append(MockTrade(price, qty))
                except: pass
    return trades

# ── Replay ────────────────────────────────────────────────────────────────────

def replay(data_dir):
    print(f"\nDaily Extreme Insider Tracker -- IMC Prosperity Round 1")
    print(f"Data: {data_dir}")
    print(f"Price tolerance: +-{PRICE_TOLERANCE} | Confirm lag: {CONFIRM_TICKS} ticks\n")

    price_ticks = load_price_ticks(data_dir)
    trade_ticks = load_trade_ticks(data_dir)

    # All unique (day, ts) combinations, sorted
    all_keys = sorted(set((d, ts) for (d, ts, _) in price_ticks))
    products  = sorted(set(prod for (_, _, prod) in price_ticks))

    print(f"Products : {', '.join(products)}")
    print(f"Total timestamps: {len(all_keys)}\n")

    # Build per-product timeline of (day, ts, mid) for forward-look
    mid_timeline = defaultdict(list)   # {product: [(day, ts, mid)]}
    for (d, ts, prod), (_, mid) in sorted(price_ticks.items()):
        mid_timeline[prod].append((d, ts, mid))

    # One tracker per product
    trackers = {prod: DailyExtremeTracker() for prod in products}

    # Signal log: {product: [(day, ts, direction, mid_at_signal)]}
    signal_log = defaultdict(list)
    # Track current signal per product to detect rising edges
    prev_signal = {prod: NEUTRAL for prod in products}

    for (day, ts) in all_keys:
        # Build state for all products at this timestamp
        depths = {}
        mkt_trades = {}
        for prod in products:
            key = (day, ts, prod)
            if key in price_ticks:
                depths[prod], _ = price_ticks[key]
            if key in trade_ticks:
                mkt_trades[prod] = trade_ticks[key]

        state = MockState(ts=ts, depths=depths, market_trades=mkt_trades)

        for prod in products:
            if prod not in depths:
                continue
            tracker = trackers[prod]
            tracker.update(state, day=day)

            sig, _ = tracker.get_signal(prod)
            # Log on rising edge (new signal or direction change)
            if sig != NEUTRAL and sig != prev_signal[prod]:
                _, mid = price_ticks[(day, ts, prod)]
                signal_log[prod].append((day, ts, sig, mid))
            prev_signal[prod] = sig

    # ── Confirmation + reporting ───────────────────────────────────────────
    print("=" * 72)
    print("  SIGNAL LOG")
    print("=" * 72)

    confirmed = defaultdict(lambda: [0, 0])   # {prod: [correct, total]}
    price_moves = defaultdict(list)

    for prod in products:
        events = signal_log[prod]
        if not events:
            print(f"\n  {prod}: no signals fired")
            continue

        timeline = mid_timeline[prod]

        print(f"\n  {prod}  ({len(events)} signals)\n")
        print(f"  {'Day':>4}  {'Timestamp':>10}  {'Dir':>6}  {'Mid':>9}  "
              f"{'Future':>9}  {'Move':>7}  {'Result':>8}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*8}")

        for (day, ts, direction, mid) in events:
            # Find mid CONFIRM_TICKS ticks later (same day preferred)
            future_ts = ts + CONFIRM_TICKS * 100
            future_mid = None
            for (fd, ft, fm) in timeline:
                if fd == day and ft >= future_ts:
                    future_mid = fm
                    break

            if future_mid is not None:
                move = future_mid - mid
                correct = (direction == LONG and move > 0) or \
                          (direction == SHORT and move < 0)
                result = "CORRECT" if correct else "WRONG"
                confirmed[prod][0] += int(correct)
                confirmed[prod][1] += 1
                price_moves[prod].append(move if direction == LONG else -move)
                move_str = f"{move:+.1f}"
            else:
                result, move_str = "no data", "   N/A"

            dir_str = "LONG " if direction == LONG else "SHORT"
            future_str = f"{future_mid:.1f}" if future_mid else "     N/A"
            print(f"  {day:>4}  {ts:>10}  {dir_str}  {mid:>9.1f}  "
                  f"{future_str:>9}  {move_str:>7}  {result:>8}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  ACCURACY SUMMARY  (confirm lag = {CONFIRM_TICKS} ticks)")
    print(f"{'='*72}\n")
    print(f"  {'Product':<30}  {'Signals':>7}  {'Confirmed':>9}  "
          f"{'Correct':>7}  {'Accuracy':>9}  Verdict")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*20}")

    for prod in products:
        n_sigs = len(signal_log[prod])
        correct, total = confirmed[prod]
        acc = correct / total if total else None

        if n_sigs == 0:
            verdict = "No signals"
        elif total < 5:
            verdict = "Too few to judge"
        elif acc >= 0.65:
            verdict = "*** INSIDER LIKELY ***"
        elif acc >= 0.55:
            verdict = "Elevated"
        else:
            verdict = "Noise"

        acc_str = f"{acc:.0%}" if acc is not None else "  N/A"
        print(f"  {prod:<30}  {n_sigs:>7}  {total:>9}  "
              f"{correct:>7}  {acc_str:>9}  {verdict}")

    # ── Verdict ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  VERDICT")
    print(f"{'='*72}")
    any_insider = False
    for prod in products:
        correct, total = confirmed[prod]
        if total >= 5:
            acc = correct / total
            if acc >= 0.65:
                any_insider = True
                moves = price_moves[prod]
                avg_move = sum(moves)/len(moves) if moves else 0
                print(f"\n  *** {prod}: INSIDER SIGNAL DETECTED ***")
                print(f"      Accuracy  : {acc:.0%} ({correct}/{total} confirmed signals correct)")
                print(f"      Avg move  : {avg_move:+.2f} price units in signal direction")
                print(f"      Strategy  : follow trades at daily running extrema")
    if not any_insider:
        print("\n  No conclusive insider signal found.")
    print()

if __name__ == "__main__":
    replay(DATA_DIR)
