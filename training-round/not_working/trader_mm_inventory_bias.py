from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json


class Trader:
    POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
    EMERALDS_FAIR = 10000

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        if "EMERALDS" in state.order_depths:
            result["EMERALDS"] = self.trade_emeralds(state, state.order_depths["EMERALDS"])
        if "TOMATOES" in state.order_depths:
            result["TOMATOES"] = self.trade_tomatoes(state, state.order_depths["TOMATOES"], data)

        trader_data = json.dumps(data, separators=(",", ":"))
        return result, 0, trader_data

    # ----------------------------
    # EMERALDS: plain anchor MM
    # ----------------------------
    def trade_emeralds(self, state: TradingState, depth: OrderDepth) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        best_bid, bid_vol, best_ask, ask_vol = book
        pos = state.position.get("EMERALDS", 0)
        limit = self.POSITION_LIMITS["EMERALDS"]
        orders: List[Order] = []

        buy_cap = limit - pos
        sell_cap = limit + pos

        # Take obvious gifts around fixed fair.
        for ask, vol in sorted(depth.sell_orders.items()):
            avail = -vol
            if ask < self.EMERALDS_FAIR and buy_cap > 0:
                qty = min(buy_cap, avail)
                if qty > 0:
                    orders.append(Order("EMERALDS", ask, qty))
                    buy_cap -= qty

        for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
            if bid > self.EMERALDS_FAIR and sell_cap > 0:
                qty = min(sell_cap, vol)
                if qty > 0:
                    orders.append(Order("EMERALDS", bid, -qty))
                    sell_cap -= qty

        # Passive quoting. Size is reduced first, price only leans a bit near limits.
        lean = 0
        if pos > 55:
            lean = -1
        elif pos < -55:
            lean = 1

        bid_px = min(best_bid + 1, self.EMERALDS_FAIR - 1 + max(lean, 0))
        ask_px = max(best_ask - 1, self.EMERALDS_FAIR + 1 + min(lean, 0))

        if bid_px >= ask_px:
            bid_px = self.EMERALDS_FAIR - 1
            ask_px = self.EMERALDS_FAIR + 1

        front = 64
        back = 16

        buy_front = min(buy_cap, max(0, front - max(0, pos)))
        sell_front = min(sell_cap, max(0, front + min(0, pos)))

        if buy_front > 0:
            orders.append(Order("EMERALDS", bid_px, buy_front))
        if sell_front > 0:
            orders.append(Order("EMERALDS", ask_px, -sell_front))

        buy_cap -= buy_front
        sell_cap -= sell_front

        if buy_cap > 0 and bid_px - 1 > 0:
            orders.append(Order("EMERALDS", bid_px - 1, min(back, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", ask_px + 1, -min(back, sell_cap)))

        return orders

    # ------------------------------------------
    # TOMATOES: passive-first with signal bias
    # ------------------------------------------
    def trade_tomatoes(self, state: TradingState, depth: OrderDepth, data: dict) -> List[Order]:
        book = self._book(depth)
        if book is None:
            return []

        best_bid, bid_vol, best_ask, ask_vol = book
        pos = state.position.get("TOMATOES", 0)
        limit = self.POSITION_LIMITS["TOMATOES"]
        orders: List[Order] = []

        buy_cap = limit - pos
        sell_cap = limit + pos

        fair = self._filtered_fair(depth)
        spoof = self._spoof_signal(depth)  # positive -> bearish, negative -> bullish

        # Smooth a fair estimate, but keep it secondary.
        prev_ema = data.get("tom_ema", fair)
        ema = 0.85 * prev_ema + 0.15 * fair
        data["tom_ema"] = ema
        fair = 0.55 * fair + 0.45 * ema

        # Use spoof as an inventory target rather than a taker signal.
        # Strong bearish spoof -> target short inventory.
        target_pos = 0
        if spoof >= 0.18:
            target_pos = -28
        elif spoof >= 0.10:
            target_pos = -14
        elif spoof <= -0.18:
            target_pos = 28
        elif spoof <= -0.10:
            target_pos = 14

        # Lean fair slightly so passive quotes sit where fills still happen.
        fair -= 4.0 * spoof

        # Opportunistic taking only when book is obviously stale versus filtered fair.
        for ask, vol in sorted(depth.sell_orders.items()):
            avail = -vol
            if ask <= fair - 4 and buy_cap > 0:
                qty = min(buy_cap, min(avail, 18))
                if qty > 0:
                    orders.append(Order("TOMATOES", ask, qty))
                    buy_cap -= qty

        for bid, vol in sorted(depth.buy_orders.items(), reverse=True):
            if bid >= fair + 4 and sell_cap > 0:
                qty = min(sell_cap, min(vol, 18))
                if qty > 0:
                    orders.append(Order("TOMATOES", bid, -qty))
                    sell_cap -= qty

        # Price placement: stay passive-first and penny jump when possible.
        base_bid = best_bid + 1
        base_ask = best_ask - 1
        if base_bid >= base_ask:
            base_bid = best_bid
            base_ask = best_ask

        # Only move prices a bit; use size as the main control.
        price_lean = 0
        if pos - target_pos > 25:
            price_lean = -1
        elif pos - target_pos < -25:
            price_lean = 1

        bid_px = base_bid + max(0, price_lean)
        ask_px = base_ask + min(0, price_lean)
        if bid_px >= ask_px:
            bid_px = base_bid
            ask_px = base_ask

        # Side sizing. Big on the preferred side, token on the weak side.
        buy_gap = target_pos - pos
        sell_gap = pos - target_pos

        strong = 66
        medium = 34
        token = 8
        back = 16

        if buy_gap >= 20:
            buy_front = strong
            sell_front = token
        elif buy_gap >= 8:
            buy_front = medium
            sell_front = token
        elif sell_gap >= 20:
            buy_front = token
            sell_front = strong
        elif sell_gap >= 8:
            buy_front = token
            sell_front = medium
        else:
            buy_front = 48
            sell_front = 48

        # Near hard limits, force the safe side harder.
        if pos >= 65:
            buy_front = 0
            sell_front = 72
            ask_px = min(ask_px, best_ask - 1 if best_ask - 1 > best_bid else best_ask)
        elif pos <= -65:
            sell_front = 0
            buy_front = 72
            bid_px = max(bid_px, best_bid + 1 if best_bid + 1 < best_ask else best_bid)

        buy_front = min(buy_cap, buy_front)
        sell_front = min(sell_cap, sell_front)

        if buy_front > 0:
            orders.append(Order("TOMATOES", bid_px, buy_front))
            buy_cap -= buy_front
        if sell_front > 0:
            orders.append(Order("TOMATOES", ask_px, -sell_front))
            sell_cap -= sell_front

        # Second layer: keep extra passive volume without burying the quote.
        if buy_cap > 0:
            orders.append(Order("TOMATOES", max(1, bid_px - 1), min(back, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("TOMATOES", ask_px + 1, -min(back, sell_cap)))

        return orders

    # ----------------------------
    # Helpers
    # ----------------------------
    def _book(self, depth: OrderDepth) -> Optional[Tuple[int, int, int, int]]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        return best_bid, depth.buy_orders[best_bid], best_ask, -depth.sell_orders[best_ask]

    def _filtered_fair(self, depth: OrderDepth) -> float:
        bid_items = sorted(depth.buy_orders.items(), reverse=True)[:3]
        ask_items = sorted(depth.sell_orders.items())[:3]

        def pick_bid(items):
            best_score = None
            best_price = items[0][0]
            for price, vol in items:
                score = vol + 0.25 * (price - items[-1][0])
                if best_score is None or score > best_score:
                    best_score = score
                    best_price = price
            return best_price

        def pick_ask(items):
            best_score = None
            best_price = items[0][0]
            worst = items[-1][0]
            for price, vol in items:
                avail = -vol
                score = avail + 0.25 * (worst - price)
                if best_score is None or score > best_score:
                    best_score = score
                    best_price = price
            return best_price

        bid_ref = pick_bid(bid_items)
        ask_ref = pick_ask(ask_items)
        return 0.5 * (bid_ref + ask_ref)

    def _spoof_signal(self, depth: OrderDepth) -> float:
        bids = sorted(depth.buy_orders.items(), reverse=True)
        asks = sorted(depth.sell_orders.items())

        bid_l23 = sum(vol for _, vol in bids[1:3]) if len(bids) > 1 else 0
        ask_l23 = sum(-vol for _, vol in asks[1:3]) if len(asks) > 1 else 0
        denom = bid_l23 + ask_l23
        if denom <= 0:
            return 0.0
        return (bid_l23 - ask_l23) / denom

    def _load(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}
