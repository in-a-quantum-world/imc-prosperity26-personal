from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import math

class Trader:
    POSITION_LIMIT = {"EMERALDS": 80, "TOMATOES": 80}

    def _best_bid_ask(self, od: OrderDepth):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def _mid(self, od: OrderDepth, fallback: float) -> float:
        bid, ask = self._best_bid_ask(od)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        if bid is not None:
            return float(bid)
        if ask is not None:
            return float(ask)
        return fallback

    def _take_edge(self, product: str, od: OrderDepth, fair: float, buy_cap: int, sell_cap: int, edge: int) -> List[Order]:
        orders: List[Order] = []
        for ask in sorted(od.sell_orders.keys()):
            vol = -od.sell_orders[ask]
            if ask <= math.floor(fair - edge) and buy_cap > 0:
                qty = min(vol, buy_cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_cap -= qty
        for bid in sorted(od.buy_orders.keys(), reverse=True):
            vol = od.buy_orders[bid]
            if bid >= math.ceil(fair + edge) and sell_cap > 0:
                qty = min(vol, sell_cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_cap -= qty
        return orders

    def _emeralds(self, od: OrderDepth, pos: int) -> List[Order]:
        product = "EMERALDS"
        limit = self.POSITION_LIMIT[product]
        fair = 10000.0
        orders: List[Order] = []
        best_bid, best_ask = self._best_bid_ask(od)
        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        orders += self._take_edge(product, od, fair, buy_cap, sell_cap, edge=1)
        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        bid_px, ask_px = 9999, 10001
        if est_pos > 50:
            bid_px, ask_px = 9998, 10000
        elif est_pos < -50:
            bid_px, ask_px = 10000, 10002

        if best_bid is not None and best_ask is not None and best_ask - best_bid >= 2:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)

        bid_sz = min(buy_cap, 42 if est_pos < 30 else 24)
        ask_sz = min(sell_cap, 42 if est_pos > -30 else 24)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))
        if buy_cap - bid_sz > 0:
            orders.append(Order(product, int(bid_px - 1), min(16, buy_cap - bid_sz)))
        if sell_cap - ask_sz > 0:
            orders.append(Order(product, int(ask_px + 1), -min(16, sell_cap - ask_sz)))
        return orders

    def _fair_tom(self, od: OrderDepth, fallback: float) -> float:
        buys = sorted(od.buy_orders.items(), reverse=True)[:3]
        sells = sorted(od.sell_orders.items())[:3]
        if buys and sells:
            bw = sum(v for _, v in buys)
            sw = sum(-v for _, v in sells)
            if bw > 0 and sw > 0:
                bid_wpx = sum(p * v for p, v in buys) / bw
                ask_wpx = sum(p * (-v) for p, v in sells) / sw
                return (bid_wpx + ask_wpx) / 2
        return self._mid(od, fallback)

    def _l23_imbalance(self, od: OrderDepth) -> float:
        buys = sorted(od.buy_orders.items(), reverse=True)
        sells = sorted(od.sell_orders.items())
        if len(buys) < 3 or len(sells) < 3:
            return 0.0
        bid_vol = sum(max(v, 0) for _, v in buys[1:3])
        ask_vol = sum(max(-v, 0) for _, v in sells[1:3])
        denom = bid_vol + ask_vol
        return 0.0 if denom == 0 else (bid_vol - ask_vol) / denom

    def run(self, state: TradingState):
        result = {}
        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == "EMERALDS":
                result[product] = self._emeralds(od, pos)
            elif product == "TOMATOES":
                limit = self.POSITION_LIMIT[product]
                fair = self._fair_tom(od, 0.0)
                imb = self._l23_imbalance(od)
                spoof = -imb
                best_bid, best_ask = self._best_bid_ask(od)
                buy_cap = max(0, limit - pos)
                sell_cap = max(0, limit + pos)
                orders = self._take_edge(product, od, fair, buy_cap, sell_cap, edge=2)
                est_pos = pos + sum(o.quantity for o in orders)
                buy_cap = max(0, limit - est_pos)
                sell_cap = max(0, limit + est_pos)

                if best_bid is None:
                    best_bid = int(round(fair)) - 3
                if best_ask is None:
                    best_ask = int(round(fair)) + 3
                spread = best_ask - best_bid
                bid_px = best_bid + 1 if spread >= 2 else best_bid
                ask_px = best_ask - 1 if spread >= 2 else best_ask

                if spoof > 0.12:
                    bid_sz, ask_sz = min(buy_cap, 64), min(sell_cap, 10)
                    ask_px += 1
                elif spoof < -0.12:
                    bid_sz, ask_sz = min(buy_cap, 10), min(sell_cap, 64)
                    bid_px -= 1
                else:
                    bid_sz, ask_sz = min(buy_cap, 38), min(sell_cap, 38)

                if bid_sz > 0:
                    orders.append(Order(product, int(bid_px), int(bid_sz)))
                if ask_sz > 0:
                    orders.append(Order(product, int(ask_px), -int(ask_sz)))
                result[product] = orders
        return result, 0, ""