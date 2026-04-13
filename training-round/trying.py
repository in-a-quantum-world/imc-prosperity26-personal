from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


class Trader:
    POSITION_LIMIT = {"EMERALDS": 80, "TOMATOES": 80}

    def _load_state(self, traderData: str) -> Dict[str, Any]:
        if traderData:
            try:
                return json.loads(traderData)
            except Exception:
                pass
        return {
            "tom_hist": [],
            "tom_ema": None,
            "tom_spoof_ema": 0.0,
        }

    def _dump_state(self, st: Dict[str, Any]) -> str:
        try:
            return json.dumps(st)
        except Exception:
            return ""

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

    def _append(self, arr: List[float], x: float, maxlen: int = 60):
        arr.append(float(x))
        if len(arr) > maxlen:
            del arr[0]

    def _mean_std(self, arr: List[float], window: int):
        if not arr:
            return 0.0, 1.0
        sub = arr[-window:] if len(arr) >= window else arr[:]
        n = len(sub)
        mean = sum(sub) / n
        var = sum((x - mean) ** 2 for x in sub) / max(1, n)
        std = math.sqrt(max(var, 1e-6))
        return mean, std

    def _l23_imbalance(self, od: OrderDepth) -> float:
        buys = sorted(od.buy_orders.items(), reverse=True)
        sells = sorted(od.sell_orders.items())
        if len(buys) < 3 or len(sells) < 3:
            return 0.0
        bid_vol = sum(max(v, 0) for _, v in buys[1:3])
        ask_vol = sum(max(-v, 0) for _, v in sells[1:3])
        denom = bid_vol + ask_vol
        if denom == 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _take_edge(
        self,
        product: str,
        od: OrderDepth,
        fair: float,
        buy_cap: int,
        sell_cap: int,
        edge: int,
    ) -> List[Order]:
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

    def _trade_emeralds(self, od: OrderDepth, pos: int) -> List[Order]:
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

        bid_px = 9999
        ask_px = 10001
        if est_pos > 50:
            bid_px, ask_px = 9998, 10000
        elif est_pos < -50:
            bid_px, ask_px = 10000, 10002

        if best_bid is not None and best_ask is not None and best_ask - best_bid >= 2:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)

        bid_sz = min(buy_cap, 44 if est_pos < 30 else 24)
        ask_sz = min(sell_cap, 44 if est_pos > -30 else 24)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)
        if rem_buy > 0:
            orders.append(Order(product, int(bid_px - 1), min(16, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(product, int(ask_px + 1), -min(16, rem_sell)))
        return orders

    def _trade_tomatoes(self, state: TradingState, od: OrderDepth, pos: int, st: Dict[str, Any]) -> List[Order]:
        product = "TOMATOES"
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        mid = self._mid(od, st["tom_ema"] if st["tom_ema"] is not None else 0.0)
        self._append(st["tom_hist"], mid, maxlen=60)

        prev_ema = st["tom_ema"]
        ema = mid if prev_ema is None else 0.94 * prev_ema + 0.06 * mid
        st["tom_ema"] = ema

        _, std_s = self._mean_std(st["tom_hist"], 14)
        z = (mid - ema) / max(std_s, 1.0)

        imb = self._l23_imbalance(od)
        spoof = -imb
        st["tom_spoof_ema"] = 0.85 * st.get("tom_spoof_ema", 0.0) + 0.15 * spoof

        fair = ema + 0.5 * st["tom_spoof_ema"]

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None:
            best_bid = int(round(fair)) - 3
        if best_ask is None:
            best_ask = int(round(fair)) + 3

        spread = best_ask - best_bid
        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        orders += self._take_edge(product, od, fair, buy_cap, sell_cap, edge=2)

        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        target = 0
        if z <= -1.4:
            target = 14
        elif z <= -0.9:
            target = 8
        elif z >= 1.4:
            target = -14
        elif z >= 0.9:
            target = -8

        if state.timestamp > 150000:
            target = int(round(target * 1.15))
        target = max(-14, min(14, target))
        diff = target - est_pos

        if spread >= 3:
            base_bid = best_bid + 1
            base_ask = best_ask - 1
            mode = "wide"
        elif spread == 2:
            base_bid = best_bid + 1
            base_ask = best_ask - 1
            mode = "normal"
        else:
            base_bid = best_bid
            base_ask = best_ask
            mode = "tight"

        bid_px = base_bid
        ask_px = base_ask

        if diff >= 8:
            bid_px = min(base_bid + 1, int(math.floor(fair)))
            ask_px = max(base_ask + 1, int(math.ceil(fair + 1)))
        elif diff >= 3:
            bid_px = min(base_bid, int(math.floor(fair)))
            ask_px = max(base_ask + 1, int(math.ceil(fair + 1)))
        elif diff <= -8:
            bid_px = min(base_bid - 1, int(math.floor(fair - 1)))
            ask_px = max(base_ask, int(math.ceil(fair)))
        elif diff <= -3:
            bid_px = min(base_bid - 1, int(math.floor(fair - 1)))
            ask_px = max(base_ask, int(math.ceil(fair)))

        spoof_bias = 0
        if st["tom_spoof_ema"] > 0.12:
            spoof_bias = 1
        elif st["tom_spoof_ema"] < -0.12:
            spoof_bias = -1

        if mode == "wide":
            base_bid_sz = 52
            base_ask_sz = 52
        elif mode == "normal":
            base_bid_sz = 40
            base_ask_sz = 40
        else:
            base_bid_sz = 18
            base_ask_sz = 18

        if diff >= 8:
            bid_sz = base_bid_sz + 10
            ask_sz = max(10, base_ask_sz - 18)
        elif diff >= 3:
            bid_sz = base_bid_sz + 6
            ask_sz = max(14, base_ask_sz - 10)
        elif diff <= -8:
            bid_sz = max(10, base_bid_sz - 18)
            ask_sz = base_ask_sz + 10
        elif diff <= -3:
            bid_sz = max(14, base_bid_sz - 10)
            ask_sz = base_ask_sz + 6
        else:
            bid_sz = base_bid_sz
            ask_sz = base_ask_sz

        if spoof_bias == 1:
            bid_sz += 4
            ask_sz = max(8, ask_sz - 4)
        elif spoof_bias == -1:
            ask_sz += 4
            bid_sz = max(8, bid_sz - 4)

        bid_sz = min(buy_cap, bid_sz)
        ask_sz = min(sell_cap, ask_sz)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)
        if mode != "tight":
            if rem_buy > 0:
                orders.append(Order(product, int(bid_px - 1), min(10, rem_buy)))
            if rem_sell > 0:
                orders.append(Order(product, int(ask_px + 1), -min(10, rem_sell)))

        return orders

    def run(self, state: TradingState):
        st = self._load_state(state.traderData)
        result: Dict[str, List[Order]] = {}
        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == "EMERALDS":
                result[product] = self._trade_emeralds(od, pos)
            elif product == "TOMATOES":
                result[product] = self._trade_tomatoes(state, od, pos, st)
        return result, 0, self._dump_state(st)