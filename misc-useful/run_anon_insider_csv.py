"""
Replay-based Anonymous Insider Detector
-----------------------------------------
Reconstructs the order book from the IMC Round 1 price CSVs and feeds each
tick into AnonymousInsiderTracker.  Reports:
  - every tick where a signal was emitted
  - whether the signal was confirmed (price moved the right way)
  - aggregate accuracy per product
  - a final verdict on whether an anonymous insider is present

Usage:
    python run_anon_insider_csv.py [path_to_data_dir]
    e.g.  python run_anon_insider_csv.py "C:/Users/rucha/Downloads/ROUND1"
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from anonymous_insider_tracker import (
    AnonymousInsiderTracker, ProductAnonState,
    LONG, SHORT, NEUTRAL,
    ROLLING_WINDOW, PRICE_TOLERANCE, ABSOLUTE_VOL_THRESHOLD,
)

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/rucha/Downloads/ROUND1"
DAYS = ["-2", "-1", "0"]

# How many ticks ahead to check for signal confirmation
CONFIRM_TICKS = 10   # 1000 ts units

# Only print signal rows with confidence above this threshold
MIN_PRINT_CONF = 0.25


# ── Mock classes (replicate Prosperity datamodel shapes) ──────────────────────

class MockOrderDepth:
    def __init__(self, buy_orders: dict, sell_orders: dict):
        self.buy_orders  = buy_orders   # {price: volume}  volume > 0
        self.sell_orders = sell_orders  # {price: volume}  volume < 0


class MockState:
    def __init__(self, timestamp: int, order_depths: dict):
        self.timestamp    = timestamp
        self.order_depths = order_depths
        self.position     = {}


# ── CSV loading ────────────────────────────────────────────────────────────────

def load_ticks(data_dir: str):
    """
    Yields (day, timestamp, product, MockOrderDepth, mid_price) in
    chronological order (day asc, timestamp asc within day).
    Only rows with at least one valid bid AND one valid ask are yielded.
    """
    rows = []
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
                    if mid <= 0:
                        continue
                    ts   = int(row['timestamp'])
                    prod = row['product'].strip()

                    # Build bid side (up to 3 levels); volume positive
                    bids = {}
                    for i in range(1, 4):
                        bp = row.get(f'bid_price_{i}', '').strip()
                        bv = row.get(f'bid_volume_{i}', '').strip()
                        if bp and bv:
                            bids[float(bp)] = int(bv)

                    # Build ask side (up to 3 levels); store volume negative
                    asks = {}
                    for i in range(1, 4):
                        ap = row.get(f'ask_price_{i}', '').strip()
                        av = row.get(f'ask_volume_{i}', '').strip()
                        if ap and av:
                            asks[float(ap)] = -int(av)

                    if not bids or not asks:
                        continue

                    depth = MockOrderDepth(bids, asks)
                    rows.append((day_num, ts, prod, depth, mid))

                except (ValueError, KeyError):
                    pass

    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    return rows


# ── Replay engine ──────────────────────────────────────────────────────────────

def replay(data_dir: str) -> None:
    print(f"\nAnonymous Insider Detector  --  IMC Prosperity Round 1")
    print(f"Data: {data_dir}")
    print(f"Params: window={ROLLING_WINDOW} ticks | tol=+-{PRICE_TOLERANCE} | "
          f"abs_vol_threshold={ABSOLUTE_VOL_THRESHOLD} | confirm_lag={CONFIRM_TICKS} ticks")

    ticks = load_ticks(data_dir)
    if not ticks:
        print("No data loaded. Check data directory.")
        return

    products = sorted({prod for _, _, prod, _, _ in ticks})
    print(f"Products: {', '.join(products)}")
    print(f"Total price ticks loaded: {len(ticks)}\n")

    # Group ticks by product for forward-look confirmation
    by_product: dict[str, list] = defaultdict(list)
    for (day, ts, prod, depth, mid) in ticks:
        by_product[prod].append((day, ts, mid))

    # Per-product tracker instance (we run one per product to keep state clean)
    trackers: dict[str, AnonymousInsiderTracker] = {
        prod: AnonymousInsiderTracker() for prod in products
    }

    # Signal log: {product: [(day, ts, direction, confidence, mid_at_signal)]}
    signal_log: dict[str, list] = defaultdict(list)

    # ── Tick-by-tick replay ────────────────────────────────────────────────
    # Group ticks by (day, ts) so all products at the same timestamp are fed
    # together (matching what the real engine would see)
    from itertools import groupby
    grouped = {}
    for (day, ts, prod, depth, mid) in ticks:
        grouped.setdefault((day, ts), {})[prod] = (depth, mid)

    for (day, ts), prod_map in sorted(grouped.items()):
        mock_depths = {prod: d for prod, (d, _) in prod_map.items()}
        state = MockState(timestamp=ts, order_depths=mock_depths)

        for prod in mock_depths:
            tracker = trackers[prod]
            prev_signal, _ = tracker.get_signal(prod)

            # Feed the tick
            tracker._current_ts = ts
            p_state = tracker._states.get(prod) or ProductAnonState(product=prod)
            if prod not in tracker._states:
                tracker._states[prod] = p_state
            _, mid = prod_map[prod]
            od = mock_depths[prod]
            new_sig, raw_conf = p_state.update(ts, mid, od)

            # Only log fresh rising-edge signals (not repeats of the same signal)
            if new_sig != NEUTRAL and new_sig != prev_signal and raw_conf >= MIN_PRINT_CONF:
                final_conf = raw_conf * p_state.hit_rate()
                signal_log[prod].append((day, ts, new_sig, final_conf, mid))

    # ── Confirmation pass ──────────────────────────────────────────────────
    print("=" * 68)
    print(f"  SIGNAL LOG  (conf >= {MIN_PRINT_CONF:.0%}, rising edge only)")
    print("=" * 68)

    confirmed_counts   = defaultdict(lambda: [0, 0])  # {prod: [correct, total]}
    signal_prices      = defaultdict(list)             # {prod: [price delta on confirm]}

    for prod in products:
        events = signal_log[prod]
        if not events:
            continue

        price_timeline = by_product[prod]  # [(day, ts, mid)]

        print(f"\n  {prod}  ({len(events)} signals)\n")
        print(f"  {'Day':>4}  {'Timestamp':>10}  {'Dir':>6}  {'Conf':>6}  "
              f"{'Mid':>9}  {'Future mid':>10}  {'Result':>8}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*6}  {'-'*6}  "
              f"{'-'*9}  {'-'*10}  {'-'*8}")

        for (day, ts, direction, conf, mid) in events:
            # Find the mid-price CONFIRM_TICKS ticks later
            future_ts = ts + CONFIRM_TICKS * 100
            future_mid = None
            for (fd, ft, fm) in price_timeline:
                if fd == day and ft >= future_ts:
                    future_mid = fm
                    break

            if future_mid is not None:
                move = future_mid - mid
                if direction == LONG:
                    correct = move > 0
                else:
                    correct = move < 0
                result  = "CORRECT" if correct else "WRONG"
                confirmed_counts[prod][0] += int(correct)
                confirmed_counts[prod][1] += 1
                signal_prices[prod].append(move if direction == LONG else -move)
            else:
                result  = "no data"

            dir_str = "LONG " if direction == LONG else "SHORT"
            future_str = f"{future_mid:>10.1f}" if future_mid is not None else "       N/A"
            print(f"  {day:>4}  {ts:>10}  {dir_str}  {conf:>6.2f}  "
                  f"{mid:>9.1f}  {future_str}  {result:>8}")

    # ── Per-product summary ────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  ACCURACY SUMMARY")
    print(f"{'='*68}\n")
    print(f"  {'Product':<30}  {'Signals':>7}  {'Correct':>7}  {'Accuracy':>9}  {'Verdict':>16}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*16}")

    for prod in products:
        total_sigs = len(signal_log[prod])
        correct, total_conf = confirmed_counts[prod]
        accuracy = correct / total_conf if total_conf else None

        if total_sigs == 0:
            verdict = "No signals"
        elif total_conf < 5:
            verdict = "Too few to judge"
        elif accuracy >= 0.65:
            verdict = "*** INSIDER LIKELY ***"
        elif accuracy >= 0.55:
            verdict = "Elevated accuracy"
        else:
            verdict = "Consistent w/ noise"

        acc_str = f"{accuracy:.0%}" if accuracy is not None else "  N/A"
        print(f"  {prod:<30}  {total_sigs:>7}  {correct:>7}  {acc_str:>9}  {verdict:>16}")

    # ── Overall verdict ────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  OVERALL VERDICT")
    print(f"{'='*68}")
    any_insider = False
    for prod in products:
        correct, total_conf = confirmed_counts[prod]
        if total_conf >= 5:
            acc = correct / total_conf
            if acc >= 0.65:
                any_insider = True
                moves = signal_prices[prod]
                avg_move = sum(moves) / len(moves) if moves else 0
                print(f"\n  *** {prod}: ANONYMOUS INSIDER DETECTED ***")
                print(f"      Signal accuracy : {acc:.0%} ({correct}/{total_conf} confirmed)")
                print(f"      Avg price move   : {avg_move:+.2f} in signal direction")
                print(f"      Interpretation   : A bot is placing anomalously large orders")
                print(f"                         at rolling price extrema with above-chance")
                print(f"                         predictive power.  Follow their orders.")
    if not any_insider:
        print("\n  No conclusive anonymous insider signal found in this dataset.")
        print("  Either no insider bot is present, or it uses a different pattern.")
        print("  Consider tuning ROLLING_WINDOW / AVG_VOLUME_MULTIPLIER.")
    print()


if __name__ == "__main__":
    replay(DATA_DIR)
