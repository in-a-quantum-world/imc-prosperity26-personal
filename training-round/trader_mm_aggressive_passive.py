from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import json
import math


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # Tuned for this wide-spread training market: maximize passive fills first,
    # keep just enough fair-value logic to avoid warehousing bad inventory.
    PARAMS = {
        "EMERALDS": {
            "anchor": 10000.0,
            "ema_alpha": 0.18,
            "micro_coeff": 1.60,
            "mean_revert_coeff": 0.10,
            "inventory_coeff": 0.04,
            "take_threshold": 1.0,
            "join_size": 58,
            "layer_size": 18,
            "inventory_soft_cap": 55,
            "max_fair_shift": 2,
        },
        "TOMATOES": {
            "anchor": None,
            "ema_alpha": 0.12,
            "micro_coeff": 1.10,
            "mean_revert_coeff": 0.12,
            "inventory_coeff": 0.05,
            "take_threshold": 1.0,
            "join_size": 54,
            "layer_size": 20,
            "inventory_soft_cap": 55,
            "max_fair_shift": 3,
        },
    }

    def run(self, state: TradingState):
        trader_state = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product in ["EMERALDS", "TOMATOES"]:
            if product in state.order_depths:
                orders, trader_state = self._trade_product(product, state, trader_state)
                result[product] = orders

        return result, 0, json.dumps(trader_state)

    def _load_state(self, trader_data: str) -> Dict:
        if not trader_data:
            return {"ema": {}, "last_mid": {}}
        try:
            loaded = json.loads(trader_data)
            loaded.setdefault("ema", {})
            loaded.setdefault("last_mid", {})
            return loaded
        except Exception:
            return {"ema": {}, "last_mid": {}}

    def _trade_product(self, product: str, state: TradingState, trader_state: Dict) -> Tuple[List[Order], Dict]:
        params = self.PARAMS[product]
        depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_orders = depth.buy_orders or {}
        sell_orders = depth.sell_orders or {}
        if not buy_orders or not sell_orders:
            return [], trader_state

        best_bid = max(buy_orders)
        best_ask = min(sell_orders)
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0

        best_bid_vol = buy_orders[best_bid]
        best_ask_vol = abs(sell_orders[best_ask])
        total_top = max(1, best_bid_vol + best_ask_vol)
        micro = (best_ask * best_bid_vol + best_bid * best_ask_vol) / total_top
        imbalance = (best_bid_vol - best_ask_vol) / total_top

        prev_ema = trader_state["ema"].get(product, mid)
        ema = params["ema_alpha"] * mid + (1.0 - params["ema_alpha"]) * prev_ema
        trader_state["ema"][product] = ema

        prev_mid = trader_state["last_mid"].get(product, mid)
        trader_state["last_mid"][product] = mid
        short_term_move = mid - prev_mid

        if params["anchor"] is not None:
            # Emeralds is anchored; use EMA only as a small adjustment.
            base_fair = 0.90 * params["anchor"] + 0.10 * ema
        else:
            base_fair = ema

        fair = base_fair
        fair += params["micro_coeff"] * (micro - mid)
        fair -= params["mean_revert_coeff"] * (mid - ema)
        fair += 0.06 * short_term_move
        fair -= params["inventory_coeff"] * position

        # Clamp how far fair can move away from displayed mid: we want more fills,
        # not a hyper-reactive model that backs off from the touch too often.
        max_shift = params["max_fair_shift"]
        fair = max(mid - max_shift, min(mid + max_shift, fair))

        orders: List[Order] = []
        buy_remaining = limit - position
        sell_remaining = limit + position

        # 1) Take only clearly stale quotes at the touch or better.
        for ask in sorted(sell_orders):
            if buy_remaining <= 0:
                break
            available = abs(sell_orders[ask])
            if ask <= fair - params["take_threshold"]:
                qty = min(buy_remaining, available)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_remaining -= qty

        for bid in sorted(buy_orders, reverse=True):
            if sell_remaining <= 0:
                break
            available = buy_orders[bid]
            if bid >= fair + params["take_threshold"]:
                qty = min(sell_remaining, available)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_remaining -= qty

        # 2) Passive quoting: penny-jump aggressively because spread is wide.
        # Default front quotes are one tick inside the touch.
        if spread >= 3:
            front_bid = best_bid + 1
            front_ask = best_ask - 1
        else:
            front_bid = best_bid
            front_ask = best_ask

        # Shift quoting by at most 1 tick using fair and imbalance, but stay near the front.
        fair_bias = int(round(fair - mid))
        if imbalance > 0.35:
            fair_bias += 1
        elif imbalance < -0.35:
            fair_bias -= 1
        fair_bias = max(-1, min(1, fair_bias))

        bid_quote = front_bid + max(0, fair_bias)
        ask_quote = front_ask + min(0, fair_bias)

        # Keep both quotes valid and inside the spread.
        bid_quote = min(bid_quote, best_ask - 1)
        ask_quote = max(ask_quote, best_bid + 1)
        if bid_quote >= ask_quote:
            bid_quote = min(front_bid, best_ask - 1)
            ask_quote = max(front_ask, best_bid + 1)
            if bid_quote >= ask_quote:
                bid_quote = best_bid
                ask_quote = best_ask

        # Inventory-aware sizing: prioritize passive fills, but size up the side
        # that reduces inventory and size down the side that worsens it.
        soft_cap = params["inventory_soft_cap"]
        inventory_pressure = 0.0 if soft_cap <= 0 else position / soft_cap
        inventory_pressure = max(-1.5, min(1.5, inventory_pressure))

        base_join = params["join_size"]
        buy_join = int(round(base_join * (1.0 - 0.45 * inventory_pressure)))
        sell_join = int(round(base_join * (1.0 + 0.45 * inventory_pressure)))
        buy_join = max(8, min(base_join + 16, buy_join))
        sell_join = max(8, min(base_join + 16, sell_join))

        # Near limits, strongly prioritize getting flatter.
        if position >= 65:
            buy_join = min(buy_join, 10)
            sell_join = max(sell_join, base_join + 10)
            bid_quote -= 1
        elif position <= -65:
            sell_join = min(sell_join, 10)
            buy_join = max(buy_join, base_join + 10)
            ask_quote += 1

        # Front layer.
        front_buy_qty = min(buy_remaining, buy_join)
        front_sell_qty = min(sell_remaining, sell_join)

        if front_buy_qty > 0:
            orders.append(Order(product, bid_quote, front_buy_qty))
            buy_remaining -= front_buy_qty
        if front_sell_qty > 0:
            orders.append(Order(product, ask_quote, -front_sell_qty))
            sell_remaining -= front_sell_qty

        # 3) Secondary layer to capture extra passive fills while keeping a backup quote.
        back_bid = max(1, bid_quote - 2)
        back_ask = ask_quote + 2

        layer_size = params["layer_size"]
        buy_back_qty = min(buy_remaining, layer_size)
        sell_back_qty = min(sell_remaining, layer_size)

        # Only place the back layer if it does not worsen an already stretched book too much.
        if buy_back_qty > 0 and position <= 70:
            orders.append(Order(product, back_bid, buy_back_qty))
            buy_remaining -= buy_back_qty
        if sell_back_qty > 0 and position >= -70:
            orders.append(Order(product, back_ask, -sell_back_qty))
            sell_remaining -= sell_back_qty

        return orders, trader_state
