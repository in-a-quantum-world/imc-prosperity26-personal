
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import jsonpickle


class Trader:
    """
    Pepper-only strategy.

    Trading idea:
    - INTARIAN_PEPPER_ROOT has near-deterministic upward drift intraday.
    - Treat it as a carry / directional product, not a symmetric MM product.
    - Get long quickly, stay long most of the day, only sell late or when absurdly rich.
    - Ignore ASH_COATED_OSMIUM entirely to isolate whether Pepper is the main PnL source.
    """

    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PEPPER = "INTARIAN_PEPPER_ROOT"

    def run(self, state: TradingState):
        result = {self.PEPPER: []}

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER])

        traderData = jsonpickle.encode({})
        return result, 0, traderData

    # ---------------- helpers ----------------

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _sorted_book(self, order_depth: OrderDepth) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        return bids, asks

    def _best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int], int, int]:
        bids, asks = self._sorted_book(order_depth)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        bid_vol = bids[0][1] if bids else 0
        ask_vol = -asks[0][1] if asks else 0
        return best_bid, best_ask, bid_vol, ask_vol

    def _mid(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _microprice(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self._mid(order_depth)
        total = bid_vol + ask_vol
        if total <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_ask * bid_vol + best_bid * ask_vol) / total

    def _spread(self, order_depth: OrderDepth) -> Optional[int]:
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def _imbalance(self, order_depth: OrderDepth) -> float:
        _, _, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _add_buy(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        qty = min(qty, limit - pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))
            pos += qty
        return pos

    def _add_sell(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        qty = min(qty, limit + pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(-qty)))
            pos -= qty
        return pos

    # ---------------- pepper strategy ----------------

    def _pepper_fair_value(self, state: TradingState, order_depth: OrderDepth) -> float:
        """
        Infer the daily anchor directly from the current book:
            FV ~= anchor + 0.001 * timestamp
        where anchor snaps to a multiple of 1000 each day.
        """
        mid = self._mid(order_depth)
        if mid is None:
            return 0.0
        raw_anchor = mid - 0.001 * state.timestamp
        anchor = 1000 * round(raw_anchor / 1000.0)
        return anchor + 0.001 * state.timestamp

    def _target_position(self, timestamp: int) -> int:
        # Hold max long for almost the full day.
        if timestamp < 920_000:
            return 80
        if timestamp < 980_000:
            return 60
        return 20

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth) -> List[Order]:
        product = self.PEPPER
        limit = self.LIMITS[product]
        pos = self._position(state, product)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        mid = self._mid(order_depth)
        spread = self._spread(order_depth)
        micro = self._microprice(order_depth)
        imb = self._imbalance(order_depth)

        if best_bid is None or best_ask is None or mid is None:
            return []

        fv = self._pepper_fair_value(state, order_depth)
        fv += 0.25 * ((micro - mid) if micro is not None else 0.0)
        fv += 0.80 * imb

        target = self._target_position(state.timestamp)
        gap = target - pos

        orders: List[Order] = []
        _, asks = self._sorted_book(order_depth)

        # 1) Aggressively cross asks while under target.
        if gap > 0:
            t = state.timestamp
            if t < 100_000:
                cross_edge = 5.0
            elif t < 400_000:
                cross_edge = 4.0
            elif t < 900_000:
                cross_edge = 3.0
            else:
                cross_edge = 2.0

            if gap >= 50:
                cross_edge += 1.0

            remaining = min(gap, limit - pos)
            for ask_price, ask_qty_signed in asks:
                ask_qty = -ask_qty_signed
                if remaining <= 0:
                    break
                if ask_price <= fv + cross_edge:
                    take = min(ask_qty, remaining)
                    if take > 0:
                        pos = self._add_buy(orders, product, ask_price, take, pos, limit)
                        remaining -= take
                else:
                    break

        # 2) Keep a strong refill bid.
        gap = target - pos
        improve_bid = 1 if spread is not None and spread >= 2 else 0
        bid_price = min(best_bid + improve_bid, int(round(fv)))
        if bid_price >= best_ask:
            bid_price = best_ask - 1

        if gap > 40:
            buy_size = 30
        elif gap > 20:
            buy_size = 20
        elif gap > 5:
            buy_size = 12
        else:
            buy_size = 4

        # 3) Sell only when forced, late, or obviously rich.
        rich = best_bid - fv
        if pos > target + 10 or rich >= 5:
            ask_price = max(best_bid + 1, min(best_ask, int(round(fv + 2))))
            sell_size = 10 if pos > target + 20 else 5
        elif pos >= 78 and spread is not None and spread >= 8:
            ask_price = best_ask
            sell_size = 2
        else:
            ask_price = max(best_ask, int(round(fv + 8)))
            sell_size = 0

        if pos >= 80:
            buy_size = 0
            if state.timestamp > 950_000:
                sell_size = max(sell_size, 12)
                ask_price = best_bid + 1
        elif pos < 20 and state.timestamp < 200_000:
            buy_size += 10

        pos = self._add_buy(orders, product, bid_price, buy_size, pos, limit)
        pos = self._add_sell(orders, product, ask_price, sell_size, pos, limit)

        return orders
