from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALD_FAIR = 10000

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            if product == "EMERALDS":
                result[product] = self.trade_emeralds(state, depth, data)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(state, depth, data)

        trader_data = json.dumps(data, separators=(",", ":"))
        return result, 0, trader_data

    # -----------------------------
    # Core product logic
    # -----------------------------
    def trade_emeralds(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        bb, bv1, ba, av1, mid = book
        pos = state.position.get("EMERALDS", 0)
        limit = self.POSITION_LIMITS["EMERALDS"]
        orders: List[Order] = []

        # EWM only used as a tiny stabilizer. EMERALDS is basically anchored.
        ema_key = "ema_emeralds"
        prev_ema = data.get(ema_key, self.EMERALD_FAIR)
        ema = 0.92 * prev_ema + 0.08 * mid
        data[ema_key] = ema

        imb1 = self._imbalance(bv1, av1)
        fair = self.EMERALD_FAIR + 1.2 * imb1 + 0.10 * (ema - self.EMERALD_FAIR)

        # Minimal stale-quote taking around the hard anchor.
        if ba <= self.EMERALD_FAIR and pos < limit:
            take = min(limit - pos, abs(depth.sell_orders.get(ba, 0)), 24)
            if take > 0:
                orders.append(Order("EMERALDS", ba, take))
                pos += take
        if bb >= self.EMERALD_FAIR and pos > -limit:
            take = min(limit + pos, depth.buy_orders.get(bb, 0), 24)
            if take > 0:
                orders.append(Order("EMERALDS", bb, -take))
                pos -= take

        orders += self._passive_mm(
            product="EMERALDS",
            best_bid=bb,
            best_ask=ba,
            fair=fair,
            pos=pos,
            limit=limit,
            timestamp=state.timestamp,
            base_size=68,
            back_size=16,
            signal=imb1,
            one_sided_threshold=0.30,
            max_bias=18,
        )
        return orders

    def trade_tomatoes(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        bb, bv1, ba, av1, mid = book
        pos = state.position.get("TOMATOES", 0)
        limit = self.POSITION_LIMITS["TOMATOES"]
        orders: List[Order] = []

        # Book features.
        bv2 = self._buy_vol(depth, 2)
        bv3 = self._buy_vol(depth, 3)
        av2 = self._sell_vol(depth, 2)
        av3 = self._sell_vol(depth, 3)

        imb1 = self._imbalance(bv1, av1)
        l23imb = self._imbalance(bv2 + bv3, av2 + av3)

        # Tutorial-round-specific read:
        # top of book imbalance is directional, but L2/L3 imbalance is often spoof-like and mean-reverts.
        # Positive l23imb (fake buy pressure) -> bearish. Negative l23imb -> bullish.
        spoof_signal = -l23imb
        combined_signal = 0.65 * imb1 + 1.35 * spoof_signal

        # Keep EMA so quotes follow the slow drift without becoming timid.
        ema_key = "ema_tomatoes"
        prev_ema = data.get(ema_key, mid)
        ema = 0.96 * prev_ema + 0.04 * mid
        data[ema_key] = ema

        mean_revert = (ema - mid) / 8.0
        fair = mid + 1.2 * combined_signal + 0.7 * mean_revert

        # Aggressive taking removed: the TOMATOES spread is ~14 ticks wide, so
        # crossing to the ask/bid costs more than the passive edge earns. Passive
        # MM only.

        # Main edge: get in front and get filled.
        orders += self._passive_mm(
            product="TOMATOES",
            best_bid=bb,
            best_ask=ba,
            fair=fair,
            pos=pos,
            limit=limit,
            timestamp=state.timestamp,
            base_size=68,
            back_size=26,
            signal=combined_signal,
            one_sided_threshold=0.34,
            max_bias=22,
        )
        return orders

    # -----------------------------
    # Passive fill engine
    # -----------------------------
    def _passive_mm(
        self,
        product: str,
        best_bid: int,
        best_ask: int,
        fair: float,
        pos: int,
        limit: int,
        timestamp: int,
        base_size: int,
        back_size: int,
        signal: float,
        one_sided_threshold: float,
        max_bias: int,
        join_inside: bool = True,
    ) -> List[Order]:
        orders: List[Order] = []
        spread = best_ask - best_bid

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)
        if buy_cap == 0 and sell_cap == 0:
            return orders

        # Quote inside the spread or join the best price.
        if join_inside and spread >= 3:
            join_bid = best_bid + 1
            join_ask = best_ask - 1
        else:
            join_bid = best_bid
            join_ask = best_ask

        # Strong inventory skew only near the limits.
        inv_ratio = pos / max(1, limit)
        inv_shift = 0
        if inv_ratio > 0.75:
            inv_shift = 3
        elif inv_ratio > 0.50:
            inv_shift = 2
        elif inv_ratio > 0.25:
            inv_shift = 1
        elif inv_ratio < -0.75:
            inv_shift = -3
        elif inv_ratio < -0.50:
            inv_shift = -2
        elif inv_ratio < -0.25:
            inv_shift = -1

        # Only a small fair-value influence. This market is won by queue priority.
        fair_shift = 0
        if fair >= (best_bid + best_ask) / 2 + 1.5:
            fair_shift = 1
        elif fair <= (best_bid + best_ask) / 2 - 1.5:
            fair_shift = -1

        directional = 0
        if signal > 0.25:
            directional = 1
        elif signal < -0.25:
            directional = -1

        bid_px = join_bid + fair_shift - max(inv_shift, 0)
        ask_px = join_ask + fair_shift - min(inv_shift, 0)

        # Keep prices valid and inside reasonable range.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            bid_px = min(join_bid, best_ask - 1)
            ask_px = max(join_ask, best_bid + 1)

        # Size asymmetry matters more than moving quotes too far.
        bias = int(max(-max_bias, min(max_bias, round(signal * 40))))
        bid_size = base_size + max(0, bias)
        ask_size = base_size + max(0, -bias)

        # Penalize the side that worsens inventory.
        if pos > 0:
            bid_size -= int(abs(pos) * 0.60)
            ask_size += int(abs(pos) * 0.20)
        elif pos < 0:
            ask_size -= int(abs(pos) * 0.60)
            bid_size += int(abs(pos) * 0.20)

        # Extra asymmetry on stronger directional read.
        if directional > 0:
            bid_size += 10
            ask_size -= 8
        elif directional < 0:
            ask_size += 10
            bid_size -= 8

        # Late session: stop dying with inventory.
        if timestamp >= 185000:
            if pos > 0:
                bid_size -= 20
                ask_size += 10
            elif pos < 0:
                ask_size -= 20
                bid_size += 10

        bid_size = max(0, min(buy_cap, bid_size))
        ask_size = max(0, min(sell_cap, ask_size))

        # In strong directional states, keep a small token quote on the weak side.
        if abs(signal) >= one_sided_threshold:
            if signal > 0:
                ask_size = min(6, ask_size)
            else:
                bid_size = min(6, bid_size)

        if bid_size > 0:
            orders.append(Order(product, int(bid_px), int(bid_size)))
        if ask_size > 0:
            orders.append(Order(product, int(ask_px), -int(ask_size)))

        # Back layer for extra passive volume.
        back_bid = max(best_bid, int(bid_px) - 1)
        back_ask = min(best_ask, int(ask_px) + 1)

        back_bid_size = max(0, min(buy_cap - bid_size, back_size))
        back_ask_size = max(0, min(sell_cap - ask_size, back_size))

        if timestamp >= 190000:
            back_bid_size = 0 if pos > 0 else back_bid_size
            back_ask_size = 0 if pos < 0 else back_ask_size

        if back_bid_size > 0 and back_bid < best_ask:
            orders.append(Order(product, int(back_bid), int(back_bid_size)))
        if back_ask_size > 0 and back_ask > best_bid:
            orders.append(Order(product, int(back_ask), -int(back_ask_size)))

        return orders

    # -----------------------------
    # Helpers
    # -----------------------------
    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _book(self, depth: OrderDepth):
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = abs(depth.sell_orders[best_ask])
        mid = (best_bid + best_ask) / 2
        return best_bid, bid_vol, best_ask, ask_vol, mid

    def _imbalance(self, buy_vol: float, sell_vol: float) -> float:
        denom = buy_vol + sell_vol
        if denom <= 0:
            return 0.0
        return (buy_vol - sell_vol) / denom

    def _buy_vol(self, depth: OrderDepth, level: int) -> int:
        prices = sorted(depth.buy_orders.keys(), reverse=True)
        if len(prices) < level:
            return 0
        return int(depth.buy_orders[prices[level - 1]])

    def _sell_vol(self, depth: OrderDepth, level: int) -> int:
        prices = sorted(depth.sell_orders.keys())
        if len(prices) < level:
            return 0
        return int(abs(depth.sell_orders[prices[level - 1]]))
