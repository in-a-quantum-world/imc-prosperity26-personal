"""
IMC Prosperity 4 — Round 3: HP + VF ONLY (no options)
Test the VF penny+OBI strategy in isolation.
"""

from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)
import json
import math
from typing import Any

HP_FAIR = 10_000
HP_SPREAD_HALF = 4
HP_ORDER_SIZE = 15
HP_MAX_POS = 200
HP_SKEW_PER_UNIT = 0.05
HP_TAKE_EDGE = 1

VF_MAX_POS = 200
VF_PENNY_SIZE = 15
VF_OBI_LEAN = 1.0
VF_INV_SKEW = 0.03
VF_TAKE_EDGE = 1

def get_mid(depth):
    if not depth.buy_orders or not depth.sell_orders:
        return None
    return (max(depth.buy_orders.keys()) + min(depth.sell_orders.keys())) / 2.0

def get_obi(depth):
    bid_vol = sum(depth.buy_orders.values()) if depth.buy_orders else 0
    ask_vol = sum(-v for v in depth.sell_orders.values()) if depth.sell_orders else 0
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0

class Trader:
    def run(self, state):
        result = {}
        conversions = 0
        positions = state.position or {}

        # HP
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hp(
                state.order_depths["HYDROGEL_PACK"],
                positions.get("HYDROGEL_PACK", 0),
            )

        # VF
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            depth = state.order_depths["VELVETFRUIT_EXTRACT"]
            mid = get_mid(depth)
            pos = positions.get("VELVETFRUIT_EXTRACT", 0)
            result["VELVETFRUIT_EXTRACT"] = self._trade_vf(depth, pos, mid)

        return result, conversions, ""

    def _trade_hp(self, depth, pos):
        orders = []
        sym = "HYDROGEL_PACK"
        mid = get_mid(depth)
        fair = HP_FAIR
        if mid is not None:
            fair = 0.7 * HP_FAIR + 0.3 * mid
        skew = -pos * HP_SKEW_PER_UNIT
        adj_fair = fair + skew

        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < adj_fair - HP_TAKE_EDGE:
                    qty = min(-depth.sell_orders[ask_p], HP_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break
        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > adj_fair + HP_TAKE_EDGE:
                    qty = min(depth.buy_orders[bid_p], HP_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        bid_p = math.floor(adj_fair - HP_SPREAD_HALF)
        ask_p = math.ceil(adj_fair + HP_SPREAD_HALF)
        bid_sz = min(HP_ORDER_SIZE, HP_MAX_POS - pos)
        ask_sz = min(HP_ORDER_SIZE, HP_MAX_POS + pos)
        if bid_sz > 0:
            orders.append(Order(sym, bid_p, bid_sz))
        if ask_sz > 0:
            orders.append(Order(sym, ask_p, -ask_sz))
        return orders

    def _trade_vf(self, depth, pos, mid):
        orders = []
        sym = "VELVETFRUIT_EXTRACT"
        if mid is None:
            return orders

        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        obi = get_obi(depth)
        obi_lean = obi * VF_OBI_LEAN
        inv_skew = -pos * VF_INV_SKEW
        adj_fair = mid + obi_lean + inv_skew

        # Aggressive takes
        if depth.sell_orders:
            for ask_p in sorted(depth.sell_orders.keys()):
                if ask_p < adj_fair - VF_TAKE_EDGE and pos < VF_MAX_POS:
                    qty = min(-depth.sell_orders[ask_p], 10, VF_MAX_POS - pos)
                    if qty > 0:
                        orders.append(Order(sym, ask_p, qty))
                        pos += qty
                else:
                    break
        if depth.buy_orders:
            for bid_p in sorted(depth.buy_orders.keys(), reverse=True):
                if bid_p > adj_fair + VF_TAKE_EDGE and pos > -VF_MAX_POS:
                    qty = min(depth.buy_orders[bid_p], 10, VF_MAX_POS + pos)
                    if qty > 0:
                        orders.append(Order(sym, bid_p, -qty))
                        pos -= qty
                else:
                    break

        # Penny quotes
        if spread >= 3:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
        else:
            our_bid = best_bid
            our_ask = best_ask

        our_bid = int(our_bid + obi_lean + inv_skew)
        our_ask = int(our_ask + obi_lean + inv_skew)
        if our_bid >= our_ask:
            our_bid = our_ask - 1

        bid_sz = min(VF_PENNY_SIZE, VF_MAX_POS - pos)
        ask_sz = min(VF_PENNY_SIZE, VF_MAX_POS + pos)
        if bid_sz > 0:
            orders.append(Order(sym, our_bid, bid_sz))
        if ask_sz > 0:
            orders.append(Order(sym, our_ask, -ask_sz))
        return orders
