"""
IMC Prosperity 4 — Round 3
Strategy 3: VELVETFRUIT_EXTRACT Mean-Reversion Scalp (isolated)
Exploits AC(1) ≈ -0.155 on tick returns — fade the last move
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
VF_MAX_POS = 200
VF_SCALP_SIZE = 8            # size per scalp order
VF_REVERT_COEFF = 0.155      # magnitude of AC(1)
VF_MIN_MOVE = 2              # only scalp if last move was >= this
VF_SPREAD_HALF = 2           # passive quote width
VF_MM_SIZE = 10              # passive MM size
VF_INVENTORY_SKEW = 0.03     # skew per unit inventory


def get_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        data = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return result, conversions, json.dumps(data)

        depth = state.order_depths["VELVETFRUIT_EXTRACT"]
        pos = (state.position or {}).get("VELVETFRUIT_EXTRACT", 0)
        mid = get_mid(depth)
        sym = "VELVETFRUIT_EXTRACT"
        orders: list[Order] = []

        if mid is None:
            return result, conversions, json.dumps(data)

        prev_mid = data.get("prev_vf_mid", None)

        # ── Scalp: fade the last tick move ──
        if prev_mid is not None:
            move = mid - prev_mid
            expected_revert = -VF_REVERT_COEFF * move

            if abs(move) >= VF_MIN_MOVE:
                if expected_revert > 0 and pos < VF_MAX_POS:
                    # Price dropped, expect bounce → buy
                    buy_price = math.floor(mid)
                    qty = min(VF_SCALP_SIZE, VF_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, buy_price, qty))

                elif expected_revert < 0 and pos > -VF_MAX_POS:
                    # Price rose, expect pullback → sell
                    sell_price = math.ceil(mid)
                    qty = min(VF_SCALP_SIZE, VF_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, sell_price, -qty))

        # ── Light passive market making to earn spread ──
        skew = -pos * VF_INVENTORY_SKEW
        bid_price = math.floor(mid - VF_SPREAD_HALF + skew)
        ask_price = math.ceil(mid + VF_SPREAD_HALF + skew)

        bid_size = min(VF_MM_SIZE, VF_MAX_POS - pos)
        ask_size = min(VF_MM_SIZE, VF_MAX_POS + pos)

        if bid_size > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        # ── Aggressive inventory unwind if too loaded ──
        if abs(pos) > VF_MAX_POS * 0.7:
            if pos > 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                dump_qty = min(depth.buy_orders[best_bid], pos // 3)
                if dump_qty > 0:
                    orders.append(Order(sym, best_bid, -dump_qty))
            elif pos < 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                dump_qty = min(-depth.sell_orders[best_ask], (-pos) // 3)
                if dump_qty > 0:
                    orders.append(Order(sym, best_ask, dump_qty))

        result[sym] = orders

        # ── Persist ──
        data["prev_vf_mid"] = mid
        return result, conversions, json.dumps(data)
