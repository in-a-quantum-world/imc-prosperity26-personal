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
            "tom_ema": None,
            "tom_hist": [],
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

    def _append(self, arr: List[float], x: float, maxlen: int = 80):
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

    def _filtered_fair_tomatoes(self, od: OrderDepth, fallback: float) -> float:
        min_sz = 12
        bid_candidates = [(p, v) for p, v in od.buy_orders.items() if v >= min_sz]
        ask_candidates = [(p, -v) for p, v in od.sell_orders.items() if -v >= min_sz]

        if bid_candidates and ask_candidates:
            bid_px, _ = max(bid_candidates, key=lambda x: (x[1], x[0]))
            ask_px, _ = max(ask_candidates, key=lambda x: (x[1], -x[0]))
            return (bid_px + ask_px) / 2

        buys = sorted(od.buy_orders.items(), reverse=True)[:3]
        sells = sorted(od.sell_orders.items())[:3]
        if buys and sells:
            bw = sum(v for _, v in buys)
            sw = sum(-v for _, v in sells)
            if bw > 0 and sw > 0:
                bid_wpx = sum(p * v for p, v in buys) / bw
                ask_wpx = sum(p * (-v) for p, v in sells) / sw
                return (bid_wpx + ask_wpx) / 2

        return fallback

    def _take_clear_edge(
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

        orders += self._take_clear_edge(product, od, fair, buy_cap, sell_cap, edge=1)

        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        bid_px = 9999
        ask_px = 10001

        if est_pos > 48:
            bid_px = 9998
            ask_px = 10000
        elif est_pos < -48:
            bid_px = 10000
            ask_px = 10002

        if best_bid is not None and best_ask is not None and best_ask - best_bid >= 2:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)

        bid_sz = min(buy_cap, 46 if est_pos < 35 else 24)
        ask_sz = min(sell_cap, 46 if est_pos > -35 else 24)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)
        if rem_buy > 0:
            orders.append(Order(product, int(bid_px - 1), min(18, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(product, int(ask_px + 1), -min(18, rem_sell)))

        return orders

    def _trade_tomatoes(self, state: TradingState, od: OrderDepth, pos: int, st: Dict[str, Any]) -> List[Order]:
        product = "TOMATOES"
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        prev_ema = st["tom_ema"]
        mid = self._mid(od, prev_ema if prev_ema is not None else 0.0)
        ema = mid if prev_ema is None else 0.95 * prev_ema + 0.05 * mid
        st["tom_ema"] = ema

        self._append(st["tom_hist"], mid, maxlen=80)
        hist = st["tom_hist"]

        mean_s, std_s = self._mean_std(hist, 12)
        mean_l, _ = self._mean_std(hist, 40)

        dev = mid - mean_s
        z = dev / max(std_s, 1.0)

        imb = self._l23_imbalance(od)
        spoof = -imb
        st["tom_spoof_ema"] = 0.85 * st.get("tom_spoof_ema", 0.0) + 0.15 * spoof

        filtered = self._filtered_fair_tomatoes(od, ema)
        fair = 0.45 * filtered + 0.35 * mean_s + 0.20 * ema + 0.5 * st["tom_spoof_ema"]

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None:
            best_bid = int(round(fair)) - 3
        if best_ask is None:
            best_ask = int(round(fair)) + 3

        spread = best_ask - best_bid
        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        orders += self._take_clear_edge(product, od, fair, buy_cap, sell_cap, edge=2)

        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        target = 0
        if z <= -1.4:
            target = 16
        elif z <= -0.9:
            target = 10
        elif z >= 1.4:
            target = -16
        elif z >= 0.9:
            target = -10

        if (mean_s - mean_l) * dev > 0:
            target = int(round(target * 0.6))

        if state.timestamp > 150000:
            target = int(round(target * 1.15))

        target = max(-16, min(16, target))
        diff = target - est_pos

        if spread >= 2:
            base_bid = best_bid + 1
            base_ask = best_ask - 1
        else:
            base_bid = best_bid
            base_ask = best_ask

        bid_px = base_bid
        ask_px = base_ask

        if diff >= 10:
            bid_px = min(base_bid + 1, int(math.floor(fair)))
            ask_px = max(base_ask + 1, int(math.ceil(fair + 1)))
        elif diff >= 4:
            bid_px = min(base_bid, int(math.floor(fair)))
            ask_px = max(base_ask + 1, int(math.ceil(fair + 1)))
        elif diff <= -10:
            bid_px = min(base_bid - 1, int(math.floor(fair - 1)))
            ask_px = max(base_ask, int(math.ceil(fair)))
        elif diff <= -4:
            bid_px = min(base_bid - 1, int(math.floor(fair - 1)))
            ask_px = max(base_ask, int(math.ceil(fair)))

        if diff >= 10:
            bid_sz = min(buy_cap, 54)
            ask_sz = min(sell_cap, 18)
        elif diff >= 4:
            bid_sz = min(buy_cap, 46)
            ask_sz = min(sell_cap, 24)
        elif diff >= -3:
            bid_sz = min(buy_cap, 40)
            ask_sz = min(sell_cap, 32)
        elif diff > -10:
            bid_sz = min(buy_cap, 28)
            ask_sz = min(sell_cap, 44)
        else:
            bid_sz = min(buy_cap, 18)
            ask_sz = min(sell_cap, 54)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)

        if diff >= 4 and rem_buy > 0:
            orders.append(Order(product, int(bid_px - 1), min(14, rem_buy)))
        elif diff <= -4 and rem_sell > 0:
            orders.append(Order(product, int(ask_px + 1), -min(14, rem_sell)))
        else:
            if rem_buy > 0:
                orders.append(Order(product, int(base_bid - 1), min(8, rem_buy)))
            if rem_sell > 0:
                orders.append(Order(product, int(base_ask + 1), -min(8, rem_sell)))

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
