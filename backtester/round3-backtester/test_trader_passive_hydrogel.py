"""Simple passive MM on HYDROGEL_PACK only.
Expected: small positive PnL, no blowup, since quotes never cross normal book."""
from datamodel import TradingState, Order


class Trader:
    def run(self, state: TradingState):
        orders = []
        depth = state.order_depths.get("HYDROGEL_PACK")
        if depth and depth.buy_orders and depth.sell_orders:
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            pos = state.position.get("HYDROGEL_PACK", 0)
            # quote one tick inside the spread
            my_bid = best_bid + 1
            my_ask = best_ask - 1
            if my_bid < my_ask:
                if pos < 40:
                    orders.append(Order("HYDROGEL_PACK", my_bid, min(10, 40 - pos)))
                if pos > -40:
                    orders.append(Order("HYDROGEL_PACK", my_ask, -min(10, 40 + pos)))
        return {"HYDROGEL_PACK": orders}, 0, ""
