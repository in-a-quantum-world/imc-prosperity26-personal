from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
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

        return result, 0, json.dumps(data, separators=(",", ":"))

    # -----------------------------
    # EMERALDS — unchanged (hard cap ~1050, not worth tuning)
    # -----------------------------
    def trade_emeralds(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        bb, bv1, ba, av1, mid = book
        pos = state.position.get("EMERALDS", 0)
        limit = self.POSITION_LIMITS["EMERALDS"]
        orders: List[Order] = []

        ema_key = "ema_emeralds"
        prev_ema = data.get(ema_key, self.EMERALD_FAIR)
        ema = 0.92 * prev_ema + 0.08 * mid
        data[ema_key] = ema

        imb1 = self._imbalance(bv1, av1)
        fair = self.EMERALD_FAIR + 1.2 * imb1 + 0.10 * (ema - self.EMERALD_FAIR)

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

    # -----------------------------
    # TOMATOES — enhanced with trend + OBI Z-score + selective taker
    # -----------------------------
    def trade_tomatoes(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        bb, bv1, ba, av1, mid = book
        pos = state.position.get("TOMATOES", 0)
        limit = self.POSITION_LIMITS["TOMATOES"]
        orders: List[Order] = []

        # --- Level-1 imbalance ---
        imb1 = self._imbalance(bv1, av1)

        # --- L2/L3 spoof signal (positive = fake buy pressure = bearish) ---
        bv2 = self._buy_vol(depth, 2)
        bv3 = self._buy_vol(depth, 3)
        av2 = self._sell_vol(depth, 2)
        av3 = self._sell_vol(depth, 3)
        l23imb = self._imbalance(bv2 + bv3, av2 + av3)
        spoof_signal = -l23imb  # positive -> bearish

        # --- EMA fair value ---
        ema_key = "ema_tomatoes"
        prev_ema = data.get(ema_key, mid)
        ema = 0.96 * prev_ema + 0.04 * mid
        data[ema_key] = ema

        # --- Trend slope: linear regression on recent 20 mids ---
        # positive slope = price trending up = bullish
        mids = data.get("tom_mids", [])
        mids.append(mid)
        if len(mids) > 20:
            mids = mids[-20:]
        data["tom_mids"] = mids

        trend_signal = 0.0
        n = len(mids)
        if n >= 8:
            x_mean = (n - 1) / 2.0
            y_mean = sum(mids) / n
            num = sum((i - x_mean) * (mids[i] - y_mean) for i in range(n))
            den = sum((i - x_mean) ** 2 for i in range(n))
            slope = num / den if den > 0 else 0.0
            # Normalise: 2 ticks/step is a strong trend → cap at ±1
            trend_signal = max(-1.0, min(1.0, slope / 2.0))

        # --- OBI Z-score: mean reversion of the imbalance itself ---
        # Extreme positive OBI (heavy buyers) → OBI will revert → slight bearish edge
        # Use Welford online update for rolling mean/variance
        obi_h = data.get("obi_hist", {"n": 0, "mean": 0.0, "m2": 0.0})
        n_obi = obi_h["n"] + 1
        delta = imb1 - obi_h["mean"]
        new_mean = obi_h["mean"] + delta / n_obi
        new_m2 = obi_h["m2"] + delta * (imb1 - new_mean)
        data["obi_hist"] = {"n": n_obi, "mean": new_mean, "m2": new_m2}

        if n_obi >= 30:
            std_obi = (new_m2 / n_obi) ** 0.5
            obi_z = (imb1 - new_mean) / max(std_obi, 0.01)
            # Clip Z to [-2, 2]; mean reversion → negative of Z
            obi_z_signal = -max(-2.0, min(2.0, obi_z)) / 2.0
        else:
            obi_z_signal = 0.0

        # --- Combined signal (bullish = positive) ---
        # Original proven weights preserved; trend and OBI-Z added as lightweight extras.
        # Avoid reducing proven weights — they were empirically calibrated.
        combined = (
            0.65 * imb1
            + 1.35 * spoof_signal
            + 0.25 * trend_signal
            + 0.15 * obi_z_signal
        )

        # --- Fair value ---
        # Trend shifts fair toward where price is heading, not just where it is.
        # A 1 tick/step slope (trend_signal ≈ 0.5) shifts fair by ~2 ticks,
        # which nudges quote placement in the trend direction.
        mean_revert = (ema - mid) / 8.0
        fair = mid + 1.2 * combined + 0.7 * mean_revert + 2.0 * trend_signal

        buy_cap = limit - pos
        sell_cap = limit + pos

        # --- Passive MM ---
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
            signal=combined,
            one_sided_threshold=0.34,
            max_bias=22,
        )
        return orders

    # -----------------------------
    # Passive fill engine (unchanged from passive_fill)
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
    ) -> List[Order]:
        orders: List[Order] = []
        spread = best_ask - best_bid

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)
        if buy_cap == 0 and sell_cap == 0:
            return orders

        if spread >= 3:
            join_bid = best_bid + 1
            join_ask = best_ask - 1
        else:
            join_bid = best_bid
            join_ask = best_ask

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

        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            bid_px = min(join_bid, best_ask - 1)
            ask_px = max(join_ask, best_bid + 1)

        bias = int(max(-max_bias, min(max_bias, round(signal * 40))))
        bid_size = base_size + max(0, bias)
        ask_size = base_size + max(0, -bias)

        if pos > 0:
            bid_size -= int(abs(pos) * 0.60)
            ask_size += int(abs(pos) * 0.20)
        elif pos < 0:
            ask_size -= int(abs(pos) * 0.60)
            bid_size += int(abs(pos) * 0.20)

        if directional > 0:
            bid_size += 10
            ask_size -= 8
        elif directional < 0:
            ask_size += 10
            bid_size -= 8

        if timestamp >= 185000:
            if pos > 0:
                bid_size -= 20
                ask_size += 10
            elif pos < 0:
                ask_size -= 20
                bid_size += 10

        bid_size = max(0, min(buy_cap, bid_size))
        ask_size = max(0, min(sell_cap, ask_size))

        if abs(signal) >= one_sided_threshold:
            if signal > 0:
                ask_size = min(6, ask_size)
            else:
                bid_size = min(6, bid_size)

        if bid_size > 0:
            orders.append(Order(product, int(bid_px), int(bid_size)))
        if ask_size > 0:
            orders.append(Order(product, int(ask_px), -int(ask_size)))

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
