from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


class Trader:
    """
    Tutorial-round strategy for EMERALDS and TOMATOES.

    Design:
    - Profit should come mostly from passive fills, not constant taking.
    - Fair value is shifted by short-horizon alpha, so this is not pure MM.
    - Inventory control is a first-class concern, especially for TOMATOES.
    - Close-to-end de-risking reduces dependence on terminal price luck.
    """

    CONFIG = {
        "EMERALDS": {
            "limit": 80,
            "base_anchor": 10000.0,
            "passive_size": 14,
            "take_size": 8,
            "soft_limit_frac": 0.55,
            "hard_limit_frac": 0.82,
            "take_threshold": 2.0,
            "inv_skew": 6.0,
            "wall_weight": 0.9,
            "short_window": 8,
            "medium_window": 40,
            "closeout_start": 965000,
            "closeout_clip": 18,
        },
        "TOMATOES": {
            "limit": 80,
            "passive_size": 10,
            "take_size": 6,
            "soft_limit_frac": 0.45,
            "hard_limit_frac": 0.72,
            "take_threshold": 3.0,
            "inv_skew": 8.0,
            "wall_weight": 1.1,
            "short_window": 10,
            "medium_window": 50,
            "closeout_start": 940000,
            "closeout_clip": 22,
        },
    }

    def __init__(self):
        self.mid_hist: Dict[str, List[float]] = {
            "EMERALDS": [],
            "TOMATOES": [],
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product not in self.CONFIG:
                continue
            position = state.position.get(product, 0)
            result[product] = self.make_orders(
                product=product,
                order_depth=order_depth,
                position=position,
                timestamp=state.timestamp,
            )

        trader_data = "tutorial_v2"
        conversions = 0
        return result, conversions, trader_data

    def make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        timestamp: int,
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        cfg = self.CONFIG[product]
        limit = cfg["limit"]
        soft_limit = int(limit * cfg["soft_limit_frac"])
        hard_limit = int(limit * cfg["hard_limit_frac"])

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol_1 = order_depth.buy_orders[best_bid]
        ask_vol_1 = -order_depth.sell_orders[best_ask]
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2

        self.mid_hist[product].append(mid)
        hist = self.mid_hist[product]

        fair = self.compute_fair_value(
            product=product,
            order_depth=order_depth,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_vol_1=bid_vol_1,
            ask_vol_1=ask_vol_1,
            mid=mid,
            hist=hist,
        )

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        orders: List[Order] = []

        inv_ratio = position / limit
        reservation_shift = cfg["inv_skew"] * inv_ratio
        fair_bid = fair - reservation_shift
        fair_ask = fair - reservation_shift

        buy_quote = None
        sell_quote = None

        # Passive quoting: penny the touch when spread allows it.
        # Otherwise only rest at the touch if the quote is still favorable.
        if spread >= 3:
            buy_quote = best_bid + 1
            sell_quote = best_ask - 1
        else:
            if best_bid <= fair_bid - 1.0:
                buy_quote = best_bid
            if best_ask >= fair_ask + 1.0:
                sell_quote = best_ask

        # Inventory-aware side suppression.
        if position >= soft_limit:
            buy_quote = None
        if position <= -soft_limit:
            sell_quote = None

        passive_size = cfg["passive_size"]
        if abs(position) >= hard_limit:
            passive_size = max(2, passive_size // 3)
        elif abs(position) >= soft_limit:
            passive_size = max(4, passive_size // 2)

        # Stronger asymmetry for TOMATOES: fight accumulation earlier.
        if product == "TOMATOES":
            if position > 0:
                passive_buy_size = max(1, passive_size // 2)
                passive_sell_size = passive_size + 2
            elif position < 0:
                passive_buy_size = passive_size + 2
                passive_sell_size = max(1, passive_size // 2)
            else:
                passive_buy_size = passive_size
                passive_sell_size = passive_size
        else:
            passive_buy_size = passive_size
            passive_sell_size = passive_size

        if buy_quote is not None and buy_capacity > 0 and buy_quote <= fair_bid + 0.5:
            qty = min(passive_buy_size, buy_capacity)
            if qty > 0:
                orders.append(Order(product, int(buy_quote), int(qty)))

        if sell_quote is not None and sell_capacity > 0 and sell_quote >= fair_ask - 0.5:
            qty = min(passive_sell_size, sell_capacity)
            if qty > 0:
                orders.append(Order(product, int(sell_quote), -int(qty)))

        # Selective taking: only when there is enough edge or urgent de-risking.
        take_size = cfg["take_size"]
        take_threshold = cfg["take_threshold"]

        if position > hard_limit:
            # Prioritize immediate de-risking.
            qty = min(take_size + 4, sell_capacity, position)
            if qty > 0:
                orders.append(Order(product, best_bid, -int(qty)))
        elif position < -hard_limit:
            qty = min(take_size + 4, buy_capacity, -position)
            if qty > 0:
                orders.append(Order(product, best_ask, int(qty)))
        else:
            if best_ask <= fair - take_threshold and buy_capacity > 0:
                qty = min(take_size, buy_capacity)
                orders.append(Order(product, best_ask, int(qty)))
            if best_bid >= fair + take_threshold and sell_capacity > 0:
                qty = min(take_size, sell_capacity)
                orders.append(Order(product, best_bid, -int(qty)))

        # End-of-day closeout: much stronger in TOMATOES to reduce mark-to-close luck.
        if timestamp >= cfg["closeout_start"]:
            clip = cfg["closeout_clip"]
            if position > 0 and sell_capacity > 0:
                qty = min(position, clip)
                if qty > 0:
                    orders.append(Order(product, best_bid, -int(qty)))
            elif position < 0 and buy_capacity > 0:
                qty = min(-position, clip)
                if qty > 0:
                    orders.append(Order(product, best_ask, int(qty)))

        return self.merge_orders(orders)

    def compute_fair_value(
        self,
        product: str,
        order_depth: OrderDepth,
        best_bid: int,
        best_ask: int,
        bid_vol_1: int,
        ask_vol_1: int,
        mid: float,
        hist: List[float],
    ) -> float:
        cfg = self.CONFIG[product]

        micro = (best_ask * bid_vol_1 + best_bid * ask_vol_1) / max(1, bid_vol_1 + ask_vol_1)

        bid_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
        ask_prices = sorted(order_depth.sell_orders.keys())

        bid_l23 = 0
        ask_l23 = 0
        if len(bid_prices) >= 3:
            bid_l23 = order_depth.buy_orders[bid_prices[1]] + order_depth.buy_orders[bid_prices[2]]
        if len(ask_prices) >= 3:
            ask_l23 = (-order_depth.sell_orders[ask_prices[1]]) + (-order_depth.sell_orders[ask_prices[2]])

        bid_wall_ratio = bid_l23 / max(1, bid_vol_1)
        ask_wall_ratio = ask_l23 / max(1, ask_vol_1)
        wall_signal = ask_wall_ratio - bid_wall_ratio

        short_n = min(len(hist), cfg["short_window"])
        med_n = min(len(hist), cfg["medium_window"])
        short_mean = sum(hist[-short_n:]) / max(1, short_n)
        med_mean = sum(hist[-med_n:]) / max(1, med_n)

        if product == "EMERALDS":
            # Stationary around a fixed anchor.
            fair = (
                0.52 * cfg["base_anchor"]
                + 0.20 * short_mean
                + 0.16 * med_mean
                + 0.12 * micro
                - cfg["wall_weight"] * wall_signal
            )
        else:
            # Short-horizon alpha, but keep it conservative to avoid overfitting trend.
            # Use mild mean reversion to the recent average plus microstructure tilt.
            fair = (
                0.42 * short_mean
                + 0.28 * med_mean
                + 0.20 * micro
                + 0.10 * mid
                - cfg["wall_weight"] * wall_signal
            )

        return fair

    def merge_orders(self, orders: List[Order]) -> List[Order]:
        by_price: Dict[int, int] = {}
        product = None
        for order in orders:
            product = order.symbol
            by_price[order.price] = by_price.get(order.price, 0) + order.quantity

        merged: List[Order] = []
        for price, qty in by_price.items():
            if qty != 0:
                merged.append(Order(product, price, qty))
        return merged
