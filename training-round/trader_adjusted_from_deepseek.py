from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any
import json
import math


class Trader:
    """
    Adjusted version of the DeepSeek idea, rewritten into a real IMC Prosperity trader.

    Key changes from the original:
    - No fake internal PnL simulation
    - No impossible cross-product "arbitrage" between EMERALDS and TOMATOES
    - Uses actual order placement against the current order book
    - Uses the two signals that the public data actually supports for TOMATOES:
        1) short-horizon mean reversion around an EMA
        2) L2/L3 imbalance (used only as a small fair-value nudge / size skew)
    - Keeps EMERALDS very simple around fair = 10000
    """

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
            "tom_imb_ema": 0.0,
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

    def _append_hist(self, arr: List[float], x: float, maxlen: int = 80):
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
        """
        Public data suggests that L2/L3 imbalance in TOMATOES is predictive,
        but not large enough to justify crossing the spread.
        So we use it only as a small nudge.
        """
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

    def _take_mispriced(
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
        if best_bid is None or best_ask is None:
            return orders

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        # obvious stale quote taking
        orders += self._take_mispriced(product, od, fair, buy_cap, sell_cap, edge=1)

        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        # EMERALDS is basically anchored at 10000 in the public data
        bid_px = 9999
        ask_px = 10001

        # inventory skew only when stretched
        if est_pos > 50:
            bid_px = 9998
            ask_px = 10000
        elif est_pos < -50:
            bid_px = 10000
            ask_px = 10002

        # penny-jump when spread allows
        if best_ask - best_bid >= 2:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)

        bid_sz = min(buy_cap, 46 if est_pos < 30 else 24)
        ask_sz = min(sell_cap, 46 if est_pos > -30 else 24)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        # second passive layer
        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)

        if rem_buy > 0:
            orders.append(Order(product, int(bid_px - 1), min(16, rem_buy)))
        if rem_sell > 0:
            orders.append(Order(product, int(ask_px + 1), -min(16, rem_sell)))

        return orders

    def _trade_tomatoes(self, od: OrderDepth, pos: int, st: Dict[str, Any], timestamp: int) -> List[Order]:
        product = "TOMATOES"
        limit = self.POSITION_LIMIT[product]
        orders: List[Order] = []

        best_bid, best_ask = self._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        mid = self._mid(od, st["tom_ema"] if st["tom_ema"] is not None else 0.0)
        self._append_hist(st["tom_hist"], mid, maxlen=80)

        # EMA anchor: simple and probably better than hardcoded "phases"
        prev_ema = st["tom_ema"]
        ema = mid if prev_ema is None else 0.93 * prev_ema + 0.07 * mid
        st["tom_ema"] = ema

        # noise estimate for the mean-reversion signal
        _, std14 = self._mean_std(st["tom_hist"], 14)
        z = (mid - ema) / max(std14, 1.0)

        # L2/L3 imbalance -> small nudge only
        imb = self._l23_imbalance(od)
        spoof_alpha = -imb
        st["tom_imb_ema"] = 0.85 * st.get("tom_imb_ema", 0.0) + 0.15 * spoof_alpha

        # fair value: mostly EMA, lightly nudged by imbalance
        fair = ema + 0.6 * st["tom_imb_ema"]

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        # small stale-quote taking, but not too aggressive
        orders += self._take_mispriced(product, od, fair, buy_cap, sell_cap, edge=2)

        est_pos = pos + sum(o.quantity for o in orders)
        buy_cap = max(0, limit - est_pos)
        sell_cap = max(0, limit + est_pos)

        spread = best_ask - best_bid

        # mean-reversion-with-noise overlay:
        # only small target positions; do not abandon the MM engine
        target = 0
        if z <= -1.4:
            target = 14
        elif z <= -0.9:
            target = 8
        elif z >= 1.4:
            target = -14
        elif z >= 0.9:
            target = -8

        # slight late-day persistence, but still modest
        if timestamp > 150000:
            target = int(round(target * 1.10))

        target = max(-14, min(14, target))
        diff = target - est_pos

        # execution posture mostly from spread width
        if spread >= 3:
            base_bid = best_bid + 1
            base_ask = best_ask - 1
            base_bid_sz = 52
            base_ask_sz = 52
        elif spread == 2:
            base_bid = best_bid + 1
            base_ask = best_ask - 1
            base_bid_sz = 40
            base_ask_sz = 40
        else:
            base_bid = best_bid
            base_ask = best_ask
            base_bid_sz = 18
            base_ask_sz = 18

        bid_px = base_bid
        ask_px = base_ask

        # only coarse 1-tick skew; size does most of the leaning
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

        # light defensive size skew from imbalance
        if st["tom_imb_ema"] > 0.12:
            bid_sz += 4
            ask_sz = max(8, ask_sz - 4)
        elif st["tom_imb_ema"] < -0.12:
            ask_sz += 4
            bid_sz = max(8, bid_sz - 4)

        bid_sz = min(buy_cap, bid_sz)
        ask_sz = min(sell_cap, ask_sz)

        if bid_sz > 0:
            orders.append(Order(product, int(bid_px), int(bid_sz)))
        if ask_sz > 0:
            orders.append(Order(product, int(ask_px), -int(ask_sz)))

        # second layer only outside the tightest state
        rem_buy = max(0, buy_cap - bid_sz)
        rem_sell = max(0, sell_cap - ask_sz)

        if spread >= 2:
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
                result[product] = self._trade_tomatoes(od, pos, st, state.timestamp)

        return result, 0, self._dump_state(st)
