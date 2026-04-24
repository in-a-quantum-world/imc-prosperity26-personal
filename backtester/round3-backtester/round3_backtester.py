"""
IMC Prosperity 4 — Round 3 specialized backtester
==================================================

Why a dedicated round-3 backtester?
-----------------------------------
Round 3 is the options round (VEV_XXXX call options on VELVETFRUIT_EXTRACT).
Traditional Prosperity backtesters disagree wildly on round 3 PnL for a few
reasons:

  1. Market trades in round 3 are very sparse (~1k/day across 10 options vs.
     ~100k price snapshots). The common "match our order against any market
     trade at equal-or-better price" rule is hugely optimistic for thin
     options where we'd never realistically cross a real taker.

  2. The standard backtester resets traderData each day. Round-3 traders need
     a fractional TTE (T = BASE - day - ts/1e6) and detect the day boundary
     by watching state.timestamp reset. Resetting traderData breaks that.

  3. End-of-run valuation of option positions using snapshot mid can be very
     noisy for deep OTM/ITM strikes whose order books are often one-sided.
     The theoretical Black-Scholes value is a more stable yardstick.

This backtester addresses those three issues:

  * Continuous 3-day sim (positions + traderData persist, timestamps reset
    at day boundary so the trader's day-detection logic works).
  * Configurable match mode — default `worse` matches market trades only at
    prices strictly better than our quote. `orderbook` disables market-trade
    fills entirely (most conservative). `all` reproduces the classic
    "same price is fine" behavior for comparison.
  * Dual end-of-run PnL: (a) mid-mark and (b) BS-model-mark for VEV options,
    so you see how much of the final number is noise vs. true theoretical
    value.

Usage
-----
    python round3_backtester.py <trader.py> \\
        --data-dir "C:/Users/rucha/Downloads" \\
        [--days 0 1 2] \\
        [--match worse] \\
        [--sigma 0.0135] \\
        [--base-tte 7]

Position limits default to HYDROGEL=50, VF=200, VEV_*=200 (educated guess; no
official source). Override with --limit PRODUCT=N (repeatable).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# datamodel lives alongside this file
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from datamodel import (
    Listing, Observation, Order, OrderDepth, Symbol, Trade, TradingState,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LIMITS = {
    "HYDROGEL_PACK": 50,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 200, "VEV_4500": 200,
    "VEV_5000": 200, "VEV_5100": 200, "VEV_5200": 200,
    "VEV_5300": 200, "VEV_5400": 200, "VEV_5500": 200,
    "VEV_6000": 200, "VEV_6500": 200,
}

VEV_PREFIX = "VEV_"


def _vev_strike(symbol: str) -> Optional[int]:
    if not symbol.startswith(VEV_PREFIX):
        return None
    try:
        return int(symbol[len(VEV_PREFIX):])
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES (for model-value end-of-run PnL)
# ─────────────────────────────────────────────────────────────────────────────

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    if x < -10: return 0.0
    if x > 10:  return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-9 or sigma <= 0:
        return max(S - K, 0.0)
    sT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sT)
    d2 = d1 - sigma * sT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PriceSnap:
    bid_prices: list[int] = field(default_factory=list)
    bid_volumes: list[int] = field(default_factory=list)
    ask_prices: list[int] = field(default_factory=list)
    ask_volumes: list[int] = field(default_factory=list)
    mid_price: float = 0.0


def load_prices(data_dir: Path, day: int) -> tuple[list[int], dict[int, dict[str, PriceSnap]]]:
    path = data_dir / f"prices_round_3_day_{day}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prices file: {path}")

    snaps: dict[int, dict[str, PriceSnap]] = defaultdict(dict)
    timestamps: set[int] = set()

    with open(path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            cols = line.rstrip("\n").split(";")
            ts = int(cols[1])
            product = cols[2]
            timestamps.add(ts)

            snap = PriceSnap(
                bid_prices=[int(x) for x in (cols[3], cols[5], cols[7]) if x],
                bid_volumes=[int(x) for x in (cols[4], cols[6], cols[8]) if x],
                ask_prices=[int(x) for x in (cols[9], cols[11], cols[13]) if x],
                ask_volumes=[int(x) for x in (cols[10], cols[12], cols[14]) if x],
                mid_price=float(cols[15]) if cols[15] else 0.0,
            )
            snaps[ts][product] = snap
    return sorted(timestamps), snaps


@dataclass
class MarketTradeEntry:
    trade: Trade
    # how much of the underlying crossing volume is still available to fill our orders
    remaining_to_fill_our_sell: int   # someone bought at trade.price, so a sell at <= trade.price can match
    remaining_to_fill_our_buy: int    # someone sold at trade.price, so a buy at >= trade.price can match


def load_trades(data_dir: Path, day: int) -> dict[int, dict[str, list[MarketTradeEntry]]]:
    path = data_dir / f"trades_round_3_day_{day}.csv"
    trades: dict[int, dict[str, list[MarketTradeEntry]]] = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return trades

    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            cols = line.rstrip("\n").split(";")
            ts = int(cols[0])
            buyer = cols[1]
            seller = cols[2]
            symbol = cols[3]
            price = int(round(float(cols[5])))
            qty = int(cols[6])
            t = Trade(symbol, price, qty, buyer, seller, ts)
            trades[ts][symbol].append(MarketTradeEntry(t, qty, qty))
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# ORDER MATCHING
# ─────────────────────────────────────────────────────────────────────────────

class MatchMode:
    ORDERBOOK = "orderbook"   # only fill against snapshot order book
    WORSE = "worse"           # + fill vs market trades at strictly better prices
    ALL = "all"               # + fill vs market trades at equal-or-better prices


def match_order_against_book(order: Order, depth: OrderDepth) -> list[tuple[int, int]]:
    """Return list of (price, volume) fills against the OrderDepth snapshot.
    Mutates depth to deduct consumed liquidity."""
    fills = []
    if order.quantity > 0:
        # buyer: match vs sell side at price <= our limit
        prices = sorted(p for p in depth.sell_orders if p <= order.price)
        for p in prices:
            avail = abs(depth.sell_orders[p])
            take = min(order.quantity, avail)
            if take <= 0:
                continue
            fills.append((p, take))
            new_remaining = avail - take
            if new_remaining == 0:
                del depth.sell_orders[p]
            else:
                depth.sell_orders[p] = -new_remaining
            order.quantity -= take
            if order.quantity == 0:
                break
    elif order.quantity < 0:
        prices = sorted((p for p in depth.buy_orders if p >= order.price), reverse=True)
        remaining = -order.quantity
        for p in prices:
            avail = depth.buy_orders[p]
            take = min(remaining, avail)
            if take <= 0:
                continue
            fills.append((p, -take))
            new_remaining = avail - take
            if new_remaining == 0:
                del depth.buy_orders[p]
            else:
                depth.buy_orders[p] = new_remaining
            remaining -= take
            if remaining == 0:
                break
        order.quantity = -remaining
    return fills


def match_order_against_trades(order: Order,
                               mt_list: list[MarketTradeEntry],
                               match_mode: str) -> list[tuple[int, int]]:
    """Fill remainder of order vs market trades.

    Match rule:
      - For a BUY  order:  eligible if trade.price < our.price  (WORSE)
                             or trade.price <= our.price         (ALL)
        The idea: trades printed BELOW our bid represent sellers who hit at
        a worse price than we were offering, so we could have been filled.
      - For a SELL order:  mirror image.
    """
    fills = []
    if order.quantity > 0:
        for mt in mt_list:
            if mt.remaining_to_fill_our_buy <= 0:
                continue
            if mt.trade.price > order.price:
                continue
            if mt.trade.price == order.price and match_mode != MatchMode.ALL:
                continue
            take = min(order.quantity, mt.remaining_to_fill_our_buy)
            if take <= 0:
                continue
            fills.append((order.price, take))  # we fill at our price (the favorable one)
            mt.remaining_to_fill_our_buy -= take
            order.quantity -= take
            if order.quantity == 0:
                break
    elif order.quantity < 0:
        remaining = -order.quantity
        for mt in mt_list:
            if mt.remaining_to_fill_our_sell <= 0:
                continue
            if mt.trade.price < order.price:
                continue
            if mt.trade.price == order.price and match_mode != MatchMode.ALL:
                continue
            take = min(remaining, mt.remaining_to_fill_our_sell)
            if take <= 0:
                continue
            fills.append((order.price, -take))
            mt.remaining_to_fill_our_sell -= take
            remaining -= take
            if remaining == 0:
                break
        order.quantity = -remaining
    return fills


# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────

class Round3BackTester:

    def __init__(self,
                 trader,
                 data_dir: Path,
                 days: list[int],
                 match_mode: str = MatchMode.WORSE,
                 limits: Optional[dict[str, int]] = None,
                 base_tte: float = 7.0,
                 sigma: float = 0.0135,
                 show_progress: bool = True,
                 print_trader: bool = False,
                 settle_at_end: bool = False):
        self.trader = trader
        self.data_dir = Path(data_dir)
        self.days = days
        self.match_mode = match_mode
        self.limits = dict(DEFAULT_LIMITS)
        if limits:
            self.limits.update(limits)
        self.base_tte = base_tte
        self.sigma = sigma
        self.show_progress = show_progress
        self.print_trader = print_trader
        self.settle_at_end = settle_at_end

        # running state (persists across days)
        self.position: dict[str, int] = defaultdict(int)
        self.cash: dict[str, float] = defaultdict(float)
        self.own_trades_prev: dict[str, list[Trade]] = defaultdict(list)
        self.trader_data: str = ""
        self.products: list[str] = []
        self.violations: list[str] = []

        # reporting
        self.ts_pnl: list[tuple[int, int, dict[str, float]]] = []  # (day, ts, pnl_by_product)
        self.final_mid: dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────

    def run(self):
        for day in self.days:
            self._run_day(day)
        return self._report()

    def _run_day(self, day: int):
        timestamps, price_snaps = load_prices(self.data_dir, day)
        market_trades = load_trades(self.data_dir, day)

        # track all products seen so far (union across days)
        for ts in timestamps[:1]:
            for prod in price_snaps[ts]:
                if prod not in self.products:
                    self.products.append(prod)

        iterator = timestamps
        if self.show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(timestamps, desc=f"day {day}", ascii=True)
            except ImportError:
                pass

        for ts in iterator:
            self._step(day, ts, price_snaps.get(ts, {}), market_trades.get(ts, {}))

        # capture last mid for each product
        last_ts = timestamps[-1]
        for prod, snap in price_snaps[last_ts].items():
            self.final_mid[prod] = snap.mid_price

    def _step(self, day: int, ts: int,
              snaps: dict[str, PriceSnap],
              mts: dict[str, list[MarketTradeEntry]]):

        # build order depths
        order_depths: dict[str, OrderDepth] = {}
        for prod, snap in snaps.items():
            od = OrderDepth()
            for p, v in zip(snap.bid_prices, snap.bid_volumes):
                od.buy_orders[p] = v
            for p, v in zip(snap.ask_prices, snap.ask_volumes):
                od.sell_orders[p] = -v  # negative = sell side
            order_depths[prod] = od

        # build market_trades view for the trader (the bot-vs-bot trades this tick)
        market_trades_view = {
            prod: [mt.trade for mt in lst]
            for prod, lst in mts.items()
        }

        state = TradingState(
            traderData=self.trader_data,
            timestamp=ts,
            listings={p: Listing(p, p, 1) for p in snaps},
            order_depths=order_depths,
            own_trades=self.own_trades_prev,
            market_trades=market_trades_view,
            position=dict(self.position),
            observations=Observation({}, {}),
        )

        # run trader
        try:
            if self.print_trader:
                orders, _conv, new_data = self.trader.run(state)
            else:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    orders, _conv, new_data = self.trader.run(state)
        except Exception as e:
            self.violations.append(f"[{day}:{ts}] trader raised {type(e).__name__}: {e}")
            orders, new_data = {}, self.trader_data

        self.trader_data = new_data or ""

        # enforce position limits — drop whole product's orders if it would breach
        sanitized: dict[str, list[Order]] = {}
        for prod, olist in (orders or {}).items():
            if not olist:
                continue
            limit = self.limits.get(prod, 200)
            pos = self.position.get(prod, 0)
            total_buy = sum(o.quantity for o in olist if o.quantity > 0)
            total_sell = sum(-o.quantity for o in olist if o.quantity < 0)
            if pos + total_buy > limit or pos - total_sell < -limit:
                self.violations.append(
                    f"[{day}:{ts}] {prod} orders breach limit {limit} "
                    f"(pos={pos}, buys={total_buy}, sells={total_sell}) — dropped"
                )
                continue
            sanitized[prod] = [Order(o.symbol, o.price, o.quantity) for o in olist]

        # match
        new_own_trades: dict[str, list[Trade]] = defaultdict(list)
        for prod, olist in sanitized.items():
            depth = order_depths.get(prod)
            mt_list = mts.get(prod, [])

            # sort so most aggressive orders match first
            buys = sorted([o for o in olist if o.quantity > 0], key=lambda o: -o.price)
            sells = sorted([o for o in olist if o.quantity < 0], key=lambda o: o.price)

            for order in buys + sells:
                if depth is not None:
                    fills = match_order_against_book(order, depth)
                    for price, vol in fills:
                        self._apply_fill(prod, price, vol, new_own_trades, ts, is_book=True)
                if order.quantity != 0 and self.match_mode != MatchMode.ORDERBOOK:
                    fills = match_order_against_trades(order, mt_list, self.match_mode)
                    for price, vol in fills:
                        self._apply_fill(prod, price, vol, new_own_trades, ts, is_book=False)

        self.own_trades_prev = new_own_trades

        # record pnl snapshot (every 100 ticks to save memory)
        if ts % 10000 == 0:
            pnl = {p: self.cash[p] + self.position[p] * snaps[p].mid_price
                   for p in snaps}
            self.ts_pnl.append((day, ts, pnl))

    def _apply_fill(self, prod: str, price: int, volume: int,
                    new_own_trades: dict, ts: int, is_book: bool):
        # volume > 0 = we bought; < 0 = we sold
        self.position[prod] += volume
        self.cash[prod] -= price * volume
        buyer = "SUBMISSION" if volume > 0 else ""
        seller = "SUBMISSION" if volume < 0 else ""
        new_own_trades[prod].append(Trade(prod, price, abs(volume), buyer, seller, ts))

    # ─────────────────────────────────────────────────────────────────

    def _report(self) -> dict:
        print("\n" + "=" * 70)
        print(f"Round 3 backtest  |  days={self.days}  |  match={self.match_mode}")
        print("=" * 70)

        vf_final = self.final_mid.get("VELVETFRUIT_EXTRACT")
        # TTE at final timestamp of last day
        last_day = self.days[-1]
        final_tte = max(self.base_tte - last_day - 0.9999, 0.001)

        mid_pnl_total = 0.0
        model_pnl_total = 0.0
        rows = []

        for prod in self.products:
            pos = self.position.get(prod, 0)
            cash = self.cash.get(prod, 0.0)
            mid = self.final_mid.get(prod, 0.0)

            # model value: for VEV use BS, else mid
            strike = _vev_strike(prod)
            if strike is not None and vf_final is not None:
                model_val = bs_call(vf_final, strike, final_tte, self.sigma)
            else:
                model_val = mid

            mid_pnl = cash + pos * mid
            model_pnl = cash + pos * model_val
            mid_pnl_total += mid_pnl
            model_pnl_total += model_pnl
            rows.append((prod, pos, cash, mid, model_val, mid_pnl, model_pnl))

        header = f"{'Product':<22} {'Pos':>6} {'Cash':>12} {'Mid':>10} {'Model':>10} {'MidPnL':>12} {'ModelPnL':>12}"
        print(header)
        print("-" * len(header))
        for prod, pos, cash, mid, model_val, mid_pnl, model_pnl in rows:
            print(f"{prod:<22} {pos:>6} {cash:>12,.0f} {mid:>10.2f} {model_val:>10.2f} "
                  f"{mid_pnl:>12,.0f} {model_pnl:>12,.0f}")
        print("-" * len(header))
        print(f"{'TOTAL':<22} {'':>6} {'':>12} {'':>10} {'':>10} "
              f"{mid_pnl_total:>12,.0f} {model_pnl_total:>12,.0f}")

        if self.violations:
            print(f"\n{len(self.violations)} violations:")
            for v in self.violations[:10]:
                print(f"  {v}")
            if len(self.violations) > 10:
                print(f"  ...and {len(self.violations) - 10} more")

        return {
            "mid_pnl_total": mid_pnl_total,
            "model_pnl_total": model_pnl_total,
            "by_product": {r[0]: {"pos": r[1], "cash": r[2], "mid": r[3],
                                  "model": r[4], "mid_pnl": r[5], "model_pnl": r[6]}
                           for r in rows},
            "violations": list(self.violations),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRADER LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_trader(trader_path: Path):
    trader_path = Path(trader_path).resolve()
    # let the trader file import datamodel from our backtester dir
    # (_THIS_DIR is already on sys.path at module import)
    spec = importlib.util.spec_from_file_location("trader_module", trader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trader at {trader_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Trader"):
        raise RuntimeError(f"{trader_path} does not expose a Trader class")
    return mod.Trader()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_limit(s: str) -> tuple[str, int]:
    k, v = s.split("=", 1)
    return k.strip(), int(v)


def main():
    ap = argparse.ArgumentParser(description="IMC Prosperity 4 round-3 backtester")
    ap.add_argument("trader", type=Path, help="path to Trader python file")
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "Downloads",
                    help="dir containing prices_round_3_day_N.csv (default: ~/Downloads)")
    ap.add_argument("--days", type=int, nargs="*", default=[0, 1, 2],
                    help="days to backtest (default: 0 1 2)")
    ap.add_argument("--match", choices=[MatchMode.ORDERBOOK, MatchMode.WORSE, MatchMode.ALL],
                    default=MatchMode.WORSE,
                    help="trade matching mode (default: worse)")
    ap.add_argument("--sigma", type=float, default=0.0135,
                    help="BS implied vol for model PnL (default: 0.0135)")
    ap.add_argument("--base-tte", type=float, default=7.0,
                    help="option TTE at day 0 start (default: 7)")
    ap.add_argument("--limit", action="append", type=_parse_limit, default=[],
                    help="override position limit, e.g. --limit VEV_5000=300 (repeatable)")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--print-trader", action="store_true",
                    help="show trader stdout (noisy)")
    args = ap.parse_args()

    trader = load_trader(args.trader)
    bt = Round3BackTester(
        trader=trader,
        data_dir=args.data_dir,
        days=args.days,
        match_mode=args.match,
        limits=dict(args.limit),
        sigma=args.sigma,
        base_tte=args.base_tte,
        show_progress=not args.no_progress,
        print_trader=args.print_trader,
    )
    bt.run()


if __name__ == "__main__":
    main()
