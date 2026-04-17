
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import jsonpickle


class Trader:
    """
    Version 6:
      - INTARIAN_PEPPER_ROOT:
          max-aggression directional carry trader
      - ASH_COATED_OSMIUM:
          higher-capacity, simplified-signal market maker inspired by Shriyans' file

    Why this version exists:
      - v5 likely captures PEPPER better, but OSMIUM was too muted
      - Shriyans' OSMIUM logic appears stronger because it is simpler and more decisive:
          * heavier weight on short-vs-long EMA
          * simple imbalance term
          * z-score reversion
          * strong-signal-only taking
          * constant take edge
          * much higher inventory capacity
    """

    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    PEPPER_SOFT_LIMIT = 80
    OSMIUM_SOFT_LIMIT = 80

    def run(self, state: TradingState):
        data = self._load_data(state)
        self._handle_day_reset(state, data)

        result = {
            self.PEPPER: [],
            self.OSMIUM: [],
        }

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER], data)

        if self.OSMIUM in state.order_depths:
            result[self.OSMIUM] = self.trade_osmium(state, state.order_depths[self.OSMIUM], data)

        traderData = jsonpickle.encode(data)
        return result, 0, traderData

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _load_data(self, state: TradingState):
        if getattr(state, "traderData", None):
            try:
                data = jsonpickle.decode(state.traderData)
                if isinstance(data, dict):
                    data.setdefault("last_timestamp", None)
                    data.setdefault("osmium_short_ema", None)
                    data.setdefault("osmium_long_ema", None)
                    data.setdefault("osmium_last_imbalance", 0.0)
                    data.setdefault("osmium_mid_hist", [])
                    data.setdefault("osmium_last_spread", None)
                    return data
            except Exception:
                pass

        return {
            "last_timestamp": None,
            "osmium_short_ema": None,
            "osmium_long_ema": None,
            "osmium_last_imbalance": 0.0,
            "osmium_mid_hist": [],
            "osmium_last_spread": None,
        }

    def _handle_day_reset(self, state: TradingState, data):
        last_ts = data.get("last_timestamp")
        if last_ts is not None and state.timestamp < last_ts:
            data["osmium_short_ema"] = None
            data["osmium_long_ema"] = None
            data["osmium_last_imbalance"] = 0.0
            data["osmium_mid_hist"] = []
            data["osmium_last_spread"] = None
        data["last_timestamp"] = state.timestamp

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def _sorted_book(self, order_depth: OrderDepth):
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        return bids, asks

    def _best_bid_ask(self, order_depth: OrderDepth):
        bids, asks = self._sorted_book(order_depth)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        bid_vol = bids[0][1] if bids else 0
        ask_vol = -asks[0][1] if asks else 0
        return best_bid, best_ask, bid_vol, ask_vol

    def _mid(self, order_depth: OrderDepth):
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _microprice(self, order_depth: OrderDepth):
        best_bid, best_ask, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return self._mid(order_depth)
        total = bid_vol + ask_vol
        if total <= 0:
            return (best_bid + best_ask) / 2.0
        return (best_ask * bid_vol + best_bid * ask_vol) / total

    def _spread(self, order_depth: OrderDepth):
        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def _imbalance(self, order_depth: OrderDepth):
        _, _, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _update_ema(self, prev, x, alpha):
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    def _append_hist(self, xs, x, cap=60):
        xs.append(x)
        if len(xs) > cap:
            del xs[0]

    def _mean_std(self, xs):
        if not xs:
            return 0.0, 1.0
        n = len(xs)
        mu = sum(xs) / n
        var = sum((x - mu) ** 2 for x in xs) / max(1, n)
        std = var ** 0.5
        return mu, max(std, 1e-6)

    def _add_buy(self, orders, product, price, qty, pos, limit):
        if qty <= 0:
            return pos
        qty = min(qty, limit - pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(qty)))
            pos += qty
        return pos

    def _add_sell(self, orders, product, price, qty, pos, limit):
        if qty <= 0:
            return pos
        qty = min(qty, limit + pos)
        if qty > 0:
            orders.append(Order(product, int(price), int(-qty)))
            pos -= qty
        return pos

    # ------------------------------------------------------------------
    # PEPPER
    # ------------------------------------------------------------------

    def _pepper_fair_value(self, state: TradingState, order_depth: OrderDepth) -> float:
        mid = self._mid(order_depth)
        if mid is None:
            return 0.0
        raw_anchor = mid - 0.001 * state.timestamp
        anchor = 1000 * round(raw_anchor / 1000.0)
        return anchor + 0.001 * state.timestamp

    def _pepper_target_position(self, state: TradingState) -> int:
        t = state.timestamp
        if t < 920_000:
            return 80
        if t < 980_000:
            return 60
        return 20

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth, data):
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

        target = self._pepper_target_position(state)
        gap = target - pos

        orders = []
        asks = sorted(order_depth.sell_orders.items())

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

    def trade_osmium(self, state: TradingState, order_depth: OrderDepth, data):
        product = self.OSMIUM
        hard_limit = self.LIMITS[product]
        soft_limit = self.OSMIUM_SOFT_LIMIT
        pos = self._position(state, product)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        mid = self._mid(order_depth)
        spread = self._spread(order_depth)
        imb = self._imbalance(order_depth)

        if None in (best_bid, best_ask, mid, spread):
            return []

        short_ema = self._update_ema(data.get("osmium_short_ema"), mid, 0.10)
        long_ema = self._update_ema(data.get("osmium_long_ema"), mid, 0.03)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        self._append_hist(data["osmium_mid_hist"], mid)
        hist = data["osmium_mid_hist"]
        mu, sigma = self._mean_std(hist[-30:] if len(hist) >= 10 else hist)
        z = (mid - mu) / sigma if sigma > 0 else 0.0

        last_imb = data.get("osmium_last_imbalance", 0.0)
        imb_change = imb - last_imb
        data["osmium_last_imbalance"] = imb
        data["osmium_last_spread"] = spread

        # simplified / stronger FV
        fv = 10000.0
        fv += 0.70 * (short_ema - long_ema)
        fv += 0.80 * imb
        fv += 0.05 * (10000.0 - mid)
        fv += -0.55 * z
        fv += -0.05 * pos

        # slight extra responsiveness from imbalance change only
        fv += 0.15 * imb_change

        if spread >= 24:
            halfwidth = 4
            base_size = 6
        elif spread >= 18:
            halfwidth = 3
            base_size = 6
        else:
            halfwidth = 2
            base_size = 5

        # high-vol filter
        if abs(z) > 2.0:
            base_size = max(3, base_size - 2)

        orders = []
        signal = fv - mid

        # strong-signal-only aggressive taking
        TAKE_THRESHOLD = 3.0
        TAKE_EDGE = 1.5

        if signal > TAKE_THRESHOLD and pos < soft_limit:
            for ask_price, ask_qty_signed in sorted(order_depth.sell_orders.items()):
                ask_qty = -ask_qty_signed
                if ask_price <= fv - TAKE_EDGE and mid <= fv:
                    take = min(ask_qty, 20, soft_limit - pos)
                    if take > 0:
                        pos = self._add_buy(orders, product, ask_price, take, pos, hard_limit)
                else:
                    break

        if signal < -TAKE_THRESHOLD and pos > -soft_limit:
            for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= fv + TAKE_EDGE and mid >= fv:
                    take = min(bid_qty, 20, soft_limit + pos)
                    if take > 0:
                        pos = self._add_sell(orders, product, bid_price, take, pos, hard_limit)
                else:
                    break

        improve_bid = 1 if signal > 0.5 and spread >= 2 else 0
        improve_ask = 1 if signal < -0.5 and spread >= 2 else 0

        bid_price = min(best_bid + improve_bid, int(round(fv - halfwidth)))
        ask_price = max(best_ask - improve_ask, int(round(fv + halfwidth)))

        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        buy_size = base_size
        sell_size = base_size

        if signal > 0.5:
            buy_size += 5
            sell_size = max(1, sell_size - 4)
        elif signal < -0.5:
            sell_size += 5
            buy_size = max(1, buy_size - 4)

        if pos > 60:
            buy_size = min(buy_size, 2)
            sell_size += 4
        elif pos < -60:
            sell_size = min(sell_size, 2)
            buy_size += 4

        buy_size = min(buy_size, max(0, hard_limit - pos))
        sell_size = min(sell_size, max(0, hard_limit + pos))

        pos = self._add_buy(orders, product, bid_price, buy_size, pos, hard_limit)
        pos = self._add_sell(orders, product, ask_price, sell_size, pos, hard_limit)

        return orders
