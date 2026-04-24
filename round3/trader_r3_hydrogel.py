"""
IMC Prosperity 4 — Round 3
Strategy 1: HYDROGEL_PACK Market Making (isolated)
Mean-reversion around 10,000, OU half-life ~200-400 ticks
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
HP_FAIR = 10_000
HP_SPREAD_HALF = 4           # passive quote width from fair
HP_ORDER_SIZE = 15
HP_MAX_POS = 200             # position limit
HP_SKEW_PER_UNIT = 0.05      # skew mid per unit of inventory
HP_TAKE_EDGE = 1             # cross spread if mispriced by more than this


def get_mid(depth: OrderDepth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        if "HYDROGEL_PACK" not in state.order_depths:
            return result, conversions, ""

        depth = state.order_depths["HYDROGEL_PACK"]
        pos = (state.position or {}).get("HYDROGEL_PACK", 0)
        orders: list[Order] = []
        sym = "HYDROGEL_PACK"

        # ── Fair value: anchor to 10k, blend with market mid ──
        mid = get_mid(depth)
        fair = HP_FAIR
        if mid is not None:
            fair = 0.7 * HP_FAIR + 0.3 * mid

        # ── Inventory skew: shift fair to shed position ──
        skew = -pos * HP_SKEW_PER_UNIT
        adj_fair = fair + skew

        # ── Aggressive takes: cross spread on clear mispricing ──
        if depth.sell_orders:
            best_ask = min(depth.sell_orders.keys())
            if best_ask < adj_fair - HP_TAKE_EDGE:
                take_qty = min(-depth.sell_orders[best_ask], HP_MAX_POS - pos)
                if take_qty > 0:
                    orders.append(Order(sym, best_ask, take_qty))
                    pos += take_qty
                # Also take L2 if available
                asks_sorted = sorted(depth.sell_orders.keys())
                for ask_p in asks_sorted[1:]:
                    if ask_p < adj_fair - HP_TAKE_EDGE:
                        qty = min(-depth.sell_orders[ask_p], HP_MAX_POS - pos)
                        if qty > 0:
                            orders.append(Order(sym, ask_p, qty))
                            pos += qty

        if depth.buy_orders:
            best_bid = max(depth.buy_orders.keys())
            if best_bid > adj_fair + HP_TAKE_EDGE:
                take_qty = min(depth.buy_orders[best_bid], HP_MAX_POS + pos)
                if take_qty > 0:
                    orders.append(Order(sym, best_bid, -take_qty))
                    pos -= take_qty
                bids_sorted = sorted(depth.buy_orders.keys(), reverse=True)
                for bid_p in bids_sorted[1:]:
                    if bid_p > adj_fair + HP_TAKE_EDGE:
                        qty = min(depth.buy_orders[bid_p], HP_MAX_POS + pos)
                        if qty > 0:
                            orders.append(Order(sym, bid_p, -qty))
                            pos -= qty

        # ── Passive quotes ──
        bid_price = math.floor(adj_fair - HP_SPREAD_HALF)
        ask_price = math.ceil(adj_fair + HP_SPREAD_HALF)

        bid_size = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        ask_size = min(HP_ORDER_SIZE, HP_MAX_POS + pos)

        if bid_size > 0:
            orders.append(Order(sym, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(sym, ask_price, -ask_size))

        result[sym] = orders
        return result, conversions, ""
