from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math

class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    POSITION_LIMIT = 100

    SKEW_FACTOR = 0.3 #coudl adjust skew later 

    #ema
    TOMATO_EMA_ALPHA = 0.05

    def __init__(self):
        self.tomato_ema: float = None

    def run(self, state: TradingState):
        result = {}
        trader_data_in = {}

        if state.traderData and state.traderData != "":
            try:
                trader_data_in = json.loads(state.traderData)
                self.tomato_ema = trader_data_in.get("tomato_ema", None)
            except Exception:
                pass

        if self.EMERALDS in state.order_depths:
            result[self.EMERALDS] = self.trade_emeralds(state)

        if self.TOMATOES in state.order_depths:
            result[self.TOMATOES] = self.trade_tomatoes(state)

        # Persist EMA across timestamps
        trader_data_out = json.dumps({"tomato_ema": self.tomato_ema})

        return result, 0, trader_data_out


    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid(self, best_bid, best_ask):
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0


    def trade_emeralds(self, state: TradingState) -> List[Order]:
        

        #spread data very consistent so passive market making makes sense

        order_depth = state.order_depths[self.EMERALDS]
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return []

        FAIR_VALUE = 10000
        position = state.position.get(self.EMERALDS, 0)
        orders = []

        
        SOFT_LIMIT = 70
        buy_cap = SOFT_LIMIT - position
        sell_cap = SOFT_LIMIT + position

        
        pos_ratio = position / SOFT_LIMIT  # ranges -1 to +1
        buy_skew = -round(pos_ratio * 2)   # shift bid down when long
        sell_skew = -round(pos_ratio * 2)  # shift ask down when long

        # Sell into any bid above fair value (someone is paying too much)
        if best_bid > FAIR_VALUE and sell_cap > 0:
            # Take all volume above fair value
            volume_to_sell = min(
                sell_cap,
                abs(order_depth.buy_orders.get(best_bid, 0))
            )
            if volume_to_sell > 0:
                orders.append(Order(self.EMERALDS, best_bid, -volume_to_sell))
                sell_cap -= volume_to_sell

        # Buy from any ask below fair value (someone selling too cheap)
        if best_ask < FAIR_VALUE and buy_cap > 0:
            volume_to_buy = min(
                buy_cap,
                abs(order_depth.sell_orders.get(best_ask, 0))
            )
            if volume_to_buy > 0:
                orders.append(Order(self.EMERALDS, best_ask, volume_to_buy))
                buy_cap -= volume_to_buy

        # Place passive orders inside the spread to capture it
        passive_buy_price = best_bid + 1 + buy_skew
        passive_sell_price = best_ask - 1 + sell_skew

        # Only post passive orders if they don't cross
        if passive_buy_price < passive_sell_price:
            if buy_cap > 0:
                orders.append(Order(self.EMERALDS, passive_buy_price, buy_cap))
            if sell_cap > 0:
                orders.append(Order(self.EMERALDS, passive_sell_price, -sell_cap))
        else:
            # Spread too tight for skewed quotes; use unbiased prices
            if buy_cap > 0:
                orders.append(Order(self.EMERALDS, best_bid + 1, buy_cap))
            if sell_cap > 0:
                orders.append(Order(self.EMERALDS, best_ask - 1, -sell_cap))

        return orders

    # ─── TOMATOES ────────────────────────────────────────────────────────────────

    def trade_tomatoes(self, state: TradingState) -> List[Order]:
        #still stable price but this time with drift
        #looks like imc prosperity3 round 1 haha
        #random walk ts so thats our drift

        order_depth = state.order_depths[self.TOMATOES]
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return []

        mid = self.get_mid(best_bid, best_ask)

        if self.tomato_ema is None:
            self.tomato_ema = mid
        else:
            self.tomato_ema = (
                self.TOMATO_EMA_ALPHA * mid
                + (1 - self.TOMATO_EMA_ALPHA) * self.tomato_ema
            )

        fair_value = self.tomato_ema
        position = state.position.get(self.TOMATOES, 0)
        orders = []

        SOFT_LIMIT = 60 #reduce drawdown?
        buy_cap = SOFT_LIMIT - position
        sell_cap = SOFT_LIMIT + position

        # Inventory skew factor
        pos_ratio = position / SOFT_LIMIT
        buy_skew = -round(pos_ratio * 2)
        sell_skew = -round(pos_ratio * 2)

        # If best_bid is above EMA fair value sell aggressively
        AGGRESSIVE_THRESHOLD = 3  # ticks above/below EMA to take aggressively
        if best_bid > fair_value + AGGRESSIVE_THRESHOLD and sell_cap > 0:
            vol = min(sell_cap, abs(order_depth.buy_orders.get(best_bid, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, best_bid, -vol))
                sell_cap -= vol

        # If best_ask belowq EMA fair value buy aggressively
        if best_ask < fair_value - AGGRESSIVE_THRESHOLD and buy_cap > 0:
            vol = min(buy_cap, abs(order_depth.sell_orders.get(best_ask, 0)))
            if vol > 0:
                orders.append(Order(self.TOMATOES, best_ask, vol))
                buy_cap -= vol

        passive_buy_price = best_bid + 1 + buy_skew
        passive_sell_price = best_ask - 1 + sell_skew

        passive_buy_price = min(passive_buy_price, int(fair_value))
        passive_sell_price = max(passive_sell_price, int(fair_value) + 1)

        if passive_buy_price < passive_sell_price:
            if buy_cap > 0:
                orders.append(Order(self.TOMATOES, passive_buy_price, buy_cap))
            if sell_cap > 0:
                orders.append(Order(self.TOMATOES, passive_sell_price, -sell_cap))
        else:
            if buy_cap > 0:
                orders.append(Order(self.TOMATOES, best_bid + 1, buy_cap))
            if sell_cap > 0:
                orders.append(Order(self.TOMATOES, best_ask - 1, -sell_cap))

        return orders