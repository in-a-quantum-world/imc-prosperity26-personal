from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json


class Trader:
    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    EMERALDS_FAIR = 10000

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

    # -------------------------
    # Product logic
    # -------------------------
    def trade_emeralds(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._top_book(depth)
        if book is None:
            return []

        best_bid, bid_vol, best_ask, ask_vol, mid = book
        pos = state.position.get("EMERALDS", 0)
        limit = self.POSITION_LIMITS["EMERALDS"]
        orders: List[Order] = []

        # Stable anchor product. Only tiny fair nudge from visible pressure.
        imb1 = self._imbalance(bid_vol, ask_vol)
        fair = self.EMERALDS_FAIR + 0.8 * imb1

        # Immediate stale taking around the hard anchor.
        if best_ask < self.EMERALDS_FAIR and pos < limit:
            take_qty = min(limit - pos, abs(depth.sell_orders.get(best_ask, 0)), 28)
            if take_qty > 0:
                orders.append(Order("EMERALDS", best_ask, take_qty))
                pos += take_qty
        if best_bid > self.EMERALDS_FAIR and pos > -limit:
            take_qty = min(limit + pos, depth.buy_orders.get(best_bid, 0), 28)
            if take_qty > 0:
                orders.append(Order("EMERALDS", best_bid, -take_qty))
                pos -= take_qty

        orders.extend(
            self._quote_passive(
                product="EMERALDS",
                depth=depth,
                fair=fair,
                signal=imb1,
                pos=pos,
                limit=limit,
                timestamp=state.timestamp,
                front_base=64,
                back_base=12,
                soft_take_width=0,
                use_filtered_fair=False,
            )
        )
        return orders

    def trade_tomatoes(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._top_book(depth)
        if book is None:
            return []

        best_bid, bid_vol, best_ask, ask_vol, mid = book
        pos = state.position.get("TOMATOES", 0)
        limit = self.POSITION_LIMITS["TOMATOES"]
        orders: List[Order] = []

        # Filtered fair from "real" size levels, inspired by Round 1 KELP style.
        fair = self._filtered_book_fair(depth, data, "TOMATOES")

        # Defensive signal only: L2/L3 spoof pressure tends to fade rather than pay to cross.
        bid23 = self._buy_vol(depth, 2) + self._buy_vol(depth, 3)
        ask23 = self._sell_vol(depth, 2) + self._sell_vol(depth, 3)
        l23_spoof = -self._imbalance(bid23, ask23)   # fake buy wall -> bearish
        imb1 = self._imbalance(bid_vol, ask_vol)

        # Small bias only. Do not let signal shove us away from the spread.
        signal = 0.35 * imb1 + 0.75 * l23_spoof
        fair += 0.8 * signal

        # Tiny aggressive takes only for obviously stale prices.
        spread = best_ask - best_bid
        if spread >= 11:
            if best_ask <= fair - 3 and signal > -0.15 and pos < limit:
                take_qty = min(limit - pos, abs(depth.sell_orders.get(best_ask, 0)), 12)
                if take_qty > 0:
                    orders.append(Order("TOMATOES", best_ask, take_qty))
                    pos += take_qty
            if best_bid >= fair + 3 and signal < 0.15 and pos > -limit:
                take_qty = min(limit + pos, depth.buy_orders.get(best_bid, 0), 12)
                if take_qty > 0:
                    orders.append(Order("TOMATOES", best_bid, -take_qty))
                    pos -= take_qty

        orders.extend(
            self._quote_passive(
                product="TOMATOES",
                depth=depth,
                fair=fair,
                signal=signal,
                pos=pos,
                limit=limit,
                timestamp=state.timestamp,
                front_base=66,
                back_base=16,
                soft_take_width=1,
                use_filtered_fair=True,
            )
        )
        return orders

    # -------------------------
    # Passive quoting engine
    # -------------------------
    def _quote_passive(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        signal: float,
        pos: int,
        limit: int,
        timestamp: int,
        front_base: int,
        back_base: int,
        soft_take_width: int,
        use_filtered_fair: bool,
    ) -> List[Order]:
        top = self._top_book(depth)
        if top is None:
            return []

        best_bid, bid_vol, best_ask, ask_vol, mid = top
        spread = best_ask - best_bid
        orders: List[Order] = []

        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)
        if buy_cap == 0 and sell_cap == 0:
            return orders

        # Default: step in front whenever spread allows.
        if spread >= 3:
            bid_px = best_bid + 1
            ask_px = best_ask - 1
        else:
            bid_px = best_bid
            ask_px = best_ask

        # Small fair-driven nudge. Never more than 1 tick.
        if fair >= mid + 1.0:
            bid_px += 1
        elif fair <= mid - 1.0:
            ask_px -= 1

        # Keep valid and avoid crossing.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)
        if bid_px >= ask_px:
            bid_px = min(best_bid + 1, best_ask - 1)
            ask_px = max(best_ask - 1, best_bid + 1)

        # Inventory handling: reduce size first, move prices second.
        inv_ratio = pos / max(1, limit)
        bid_size = front_base
        ask_size = front_base

        # Side sizing from signal: bigger on the safer side, token on the toxic side.
        if signal > 0.12:
            bid_size += 10
            ask_size -= 8
        elif signal < -0.12:
            ask_size += 10
            bid_size -= 8

        # Inventory skew: keep quoting, but lean out near limits.
        if pos > 0:
            bid_size -= int(0.55 * pos)
            ask_size += int(0.10 * pos)
        elif pos < 0:
            ask_size -= int(0.55 * abs(pos))
            bid_size += int(0.10 * abs(pos))

        # Only shift quote prices once inventory is actually stretched.
        if inv_ratio > 0.70:
            bid_px -= 2
            ask_px -= 1
        elif inv_ratio > 0.45:
            bid_px -= 1
        elif inv_ratio < -0.70:
            bid_px += 1
            ask_px += 2
        elif inv_ratio < -0.45:
            ask_px += 1

        # Endgame: stop relying on lucky terminal inventory.
        if timestamp >= 185000:
            if pos > 0:
                bid_size -= 18
                ask_size += 8
                ask_px = max(ask_px - 1, best_bid + 1)
            elif pos < 0:
                ask_size -= 18
                bid_size += 8
                bid_px = min(bid_px + 1, best_ask - 1)

        # Stay inside legal book range.
        bid_px = min(int(bid_px), best_ask - 1)
        ask_px = max(int(ask_px), best_bid + 1)

        bid_size = max(0, min(buy_cap, bid_size))
        ask_size = max(0, min(sell_cap, ask_size))

        # Do not vanish entirely on one side, but cut the weak side to a token quote.
        if abs(signal) > 0.22:
            if signal > 0:
                ask_size = min(8, ask_size)
            else:
                bid_size = min(8, bid_size)

        if bid_size > 0 and bid_px < best_ask:
            orders.append(Order(product, bid_px, int(bid_size)))
        if ask_size > 0 and ask_px > best_bid:
            orders.append(Order(product, ask_px, -int(ask_size)))

        # Back layer for extra passive fills.
        rem_buy = max(0, buy_cap - bid_size)
        rem_sell = max(0, sell_cap - ask_size)

        back_bid_px = max(best_bid, bid_px - 1)
        back_ask_px = min(best_ask, ask_px + 1)

        back_bid_size = min(rem_buy, back_base)
        back_ask_size = min(rem_sell, back_base)

        if timestamp >= 190000:
            if pos > 0:
                back_bid_size = 0
            elif pos < 0:
                back_ask_size = 0

        if back_bid_size > 0 and back_bid_px < best_ask:
            orders.append(Order(product, int(back_bid_px), int(back_bid_size)))
        if back_ask_size > 0 and back_ask_px > best_bid:
            orders.append(Order(product, int(back_ask_px), -int(back_ask_size)))

        # Optional zero-edge flattening at the fair anchor if inventory is stretched.
        if abs(pos) >= 58:
            if product == "EMERALDS":
                flatten_px = self.EMERALDS_FAIR
            else:
                flatten_px = round(fair)

            if pos > 0 and sell_cap > 0:
                px = max(best_bid + 1, min(best_ask, int(flatten_px)))
                if px > best_bid:
                    orders.append(Order(product, px, -min(10, sell_cap)))
            elif pos < 0 and buy_cap > 0:
                px = min(best_ask - 1, max(best_bid, int(flatten_px)))
                if px < best_ask:
                    orders.append(Order(product, px, min(10, buy_cap)))

        return orders

    # -------------------------
    # Book helpers
    # -------------------------
    def _filtered_book_fair(self, depth: OrderDepth, data: dict, product: str) -> float:
        top = self._top_book(depth)
        if top is None:
            return 0.0
        best_bid, _, best_ask, _, mid = top

        bid_candidates: List[Tuple[int, int]] = []
        ask_candidates: List[Tuple[int, int]] = []
        for level in [1, 2, 3]:
            bp = self._buy_price(depth, level)
            bv = self._buy_vol(depth, level)
            ap = self._sell_price(depth, level)
            av = self._sell_vol(depth, level)
            if bp is not None and bv > 0:
                bid_candidates.append((bv, bp))
            if ap is not None and av > 0:
                ask_candidates.append((av, ap))

        if not bid_candidates or not ask_candidates:
            return mid

        # Main estimate: midpoint of the max-volume levels.
        vol_bid_px = max(bid_candidates, key=lambda x: (x[0], x[1]))[1]
        vol_ask_px = max(ask_candidates, key=lambda x: (x[0], -x[1]))[1]
        wall_mid = (vol_bid_px + vol_ask_px) / 2

        # Tiny EMA to reduce one-tick flicker.
        ema_key = f"ema_{product}"
        prev = data.get(ema_key, wall_mid)
        ema = 0.85 * prev + 0.15 * wall_mid
        data[ema_key] = ema
        return 0.65 * wall_mid + 0.35 * ema

    def _top_book(self, depth: OrderDepth) -> Optional[Tuple[int, int, int, int, float]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        bid_vol = int(depth.buy_orders[best_bid])
        ask_vol = int(abs(depth.sell_orders[best_ask]))
        mid = (best_bid + best_ask) / 2
        return best_bid, bid_vol, best_ask, ask_vol, mid

    def _imbalance(self, buy_vol: float, sell_vol: float) -> float:
        denom = buy_vol + sell_vol
        if denom <= 0:
            return 0.0
        return (buy_vol - sell_vol) / denom

    def _buy_price(self, depth: OrderDepth, level: int) -> Optional[int]:
        prices = sorted(depth.buy_orders.keys(), reverse=True)
        if len(prices) < level:
            return None
        return int(prices[level - 1])

    def _sell_price(self, depth: OrderDepth, level: int) -> Optional[int]:
        prices = sorted(depth.sell_orders.keys())
        if len(prices) < level:
            return None
        return int(prices[level - 1])

    def _buy_vol(self, depth: OrderDepth, level: int) -> int:
        px = self._buy_price(depth, level)
        if px is None:
            return 0
        return int(depth.buy_orders[px])

    def _sell_vol(self, depth: OrderDepth, level: int) -> int:
        px = self._sell_price(depth, level)
        if px is None:
            return 0
        return int(abs(depth.sell_orders[px]))

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}
