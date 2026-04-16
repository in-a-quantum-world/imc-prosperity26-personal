
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import math
import jsonpickle


class Trader:
    """
    Round 1 trader for:
      - INTARIAN_PEPPER_ROOT
      - ASH_COATED_OSMIUM

    Design:
      1) INTARIAN_PEPPER_ROOT:
         - Uses the near-deterministic drift fair value:
              FV ~= day_anchor + 0.001 * timestamp
         - day_anchor is inferred from the current book and smoothed.
         - Strategy is bid-dominant / drift-following market making.
         - It will still sell when too long or when the book is clearly rich.

      2) ASH_COATED_OSMIUM:
         - Anchored around 10000.
         - Uses microprice + short-vs-long EMA deviation to track the slow wave /
           persistence without overfitting a fragile explicit sine phase model.
         - Mean-reverting around 10000, but willing to lean with the short-term move.
         - Spread-regime aware: more active in wider spreads.

    Output format:
      return result, conversions, traderData
    """

    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    def run(self, state: TradingState):
        data = self._load_data(state)

        result: Dict[str, List[Order]] = {
            self.PEPPER: [],
            self.OSMIUM: [],
        }

        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER], data)

        if self.OSMIUM in state.order_depths:
            result[self.OSMIUM] = self.trade_osmium(state, state.order_depths[self.OSMIUM], data)

        traderData = jsonpickle.encode(data)
        conversions = 0
        return result, conversions, traderData

    # -------------------------------------------------------------------------
    # Persistent state
    # -------------------------------------------------------------------------

    def _load_data(self, state: TradingState) -> Dict:
        if getattr(state, "traderData", None):
            try:
                data = jsonpickle.decode(state.traderData)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        return {
            "pepper_anchor": None,
            "osmium_short_ema": None,
            "osmium_long_ema": None,
            "last_mid": {
                self.PEPPER: None,
                self.OSMIUM: None,
            },
        }

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

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

    def _l1_imbalance(self, order_depth: OrderDepth) -> float:
        _, _, bid_vol, ask_vol = self._best_bid_ask(order_depth)
        denom = bid_vol + ask_vol
        if denom <= 0:
            return 0.0
        return (bid_vol - ask_vol) / denom

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _round_to_tick(self, x: float) -> int:
        return int(round(x))

    def _update_ema(self, prev: Optional[float], x: float, alpha: float) -> float:
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    # -------------------------------------------------------------------------
    # Order placement helpers
    # -------------------------------------------------------------------------

    def _add_buy(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        allowed = max(0, limit - pos)
        q = min(qty, allowed)
        if q > 0:
            orders.append(Order(product, int(price), int(q)))
            pos += q
        return pos

    def _add_sell(self, orders: List[Order], product: str, price: int, qty: int, pos: int, limit: int) -> int:
        if qty <= 0:
            return pos
        allowed = max(0, limit + pos)
        q = min(qty, allowed)
        if q > 0:
            orders.append(Order(product, int(price), int(-q)))
            pos -= q
        return pos

    def _take_crossed_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        pos: int,
        limit: int,
        buy_edge: float,
        sell_edge: float,
        max_take_per_side: int,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        bids, asks = self._sorted_book(order_depth)

        bought = 0
        for ask_price, ask_qty_signed in asks:
            ask_qty = -ask_qty_signed
            if ask_price <= fair_value - buy_edge and bought < max_take_per_side:
                take_qty = min(ask_qty, max_take_per_side - bought, limit - pos)
                if take_qty > 0:
                    orders.append(Order(product, ask_price, take_qty))
                    pos += take_qty
                    bought += take_qty
            else:
                break

        sold = 0
        for bid_price, bid_qty in bids:
            if bid_price >= fair_value + sell_edge and sold < max_take_per_side:
                take_qty = min(bid_qty, max_take_per_side - sold, limit + pos)
                if take_qty > 0:
                    orders.append(Order(product, bid_price, -take_qty))
                    pos -= take_qty
                    sold += take_qty
            else:
                break

        return orders, pos

    # -------------------------------------------------------------------------
    # INTARIAN_PEPPER_ROOT
    # -------------------------------------------------------------------------

    def trade_pepper(self, state: TradingState, order_depth: OrderDepth, data: Dict) -> List[Order]:
        product = self.PEPPER
        limit = self.LIMITS[product]
        pos = self._position(state, product)

        mid = self._mid(order_depth)
        micro = self._microprice(order_depth)
        spread = self._spread(order_depth)
        imbalance = self._l1_imbalance(order_depth)

        if mid is None:
            return []

        # Infer the day anchor:
        #   mid - 0.001 * timestamp should be close to a multiple of 1000.
        raw_anchor = mid - 0.001 * state.timestamp
        snapped_anchor = 1000 * round(raw_anchor / 1000.0)

        prev_anchor = data.get("pepper_anchor")
        if prev_anchor is None:
            anchor = snapped_anchor
        else:
            # Keep the anchor sticky intraday, but allow reset if the market clearly shifts by ~1000.
            if abs(snapped_anchor - prev_anchor) >= 700:
                anchor = snapped_anchor
            else:
                anchor = 0.90 * prev_anchor + 0.10 * snapped_anchor

        data["pepper_anchor"] = anchor

        drift_fv = anchor + 0.001 * state.timestamp
        micro_tilt = 0.80 * (micro - mid) if micro is not None else 0.0
        imbalance_tilt = 2.50 * imbalance
        inventory_tilt = -0.10 * pos

        fair_value = drift_fv + micro_tilt + imbalance_tilt + inventory_tilt

        orders: List[Order] = []

        # Step 1: take obvious gifts around fair value.
        take_orders, pos = self._take_crossed_orders(
            product=product,
            order_depth=order_depth,
            fair_value=fair_value,
            pos=pos,
            limit=limit,
            buy_edge=2.0,     # Pepper residual noise is small, so 2 ticks is already meaningful
            sell_edge=2.0,
            max_take_per_side=18,
        )
        orders.extend(take_orders)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        signal = fair_value - mid

        # Regime:
        #   neutral / medium / strong
        abs_signal = abs(signal)

        if abs_signal >= 3.0:
            regime = "strong"
        elif abs_signal >= 1.2:
            regime = "medium"
        else:
            regime = "neutral"

        # Base widths
        if spread >= 12:
            buy_offset = 1
            sell_offset = 2
        else:
            buy_offset = 0
            sell_offset = 1

        # Pepper is drift-up product -> default bid dominance unless inventory is very long.
        long_pressure = pos / limit

        buy_bias = 0.0
        if long_pressure < 0.50:
            buy_bias += 0.8
        if long_pressure < 0.20:
            buy_bias += 0.6
        if signal > 0:
            buy_bias += 0.7
        if imbalance > 0.15:
            buy_bias += 0.5

        sell_bias = 0.0
        if long_pressure > 0.60:
            sell_bias += 1.0
        if long_pressure > 0.80:
            sell_bias += 0.8
        if signal < 0:
            sell_bias += 0.5

        # Improve by one tick when conviction is stronger and spread permits.
        bid_price = min(best_bid + (1 if (regime != "neutral" and spread >= 3) else 0),
                        self._round_to_tick(fair_value - buy_offset + buy_bias * 0.3))
        ask_price = max(best_ask - (1 if (regime == "strong" and signal < -2 and spread >= 3) else 0),
                        self._round_to_tick(fair_value + sell_offset + sell_bias * 0.3))

        # Keep quotes sane.
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        # Size logic: larger on favored side, smaller token quote on the other.
        base_buy = 16 if spread >= 10 else 10
        base_sell = 8 if spread >= 10 else 5

        if regime == "strong":
            buy_size = base_buy + 10 if signal > 0 else 6
            sell_size = 5 if signal > 0 else base_sell + 8
        elif regime == "medium":
            buy_size = base_buy + 5 if signal > 0 else 7
            sell_size = 6 if signal > 0 else base_sell + 4
        else:
            buy_size = base_buy
            sell_size = base_sell

        # Inventory overrides
        if pos > 55:
            buy_size = min(buy_size, 3)
            sell_size += 10
            ask_price = max(best_bid + 1, min(ask_price, best_ask))
        elif pos > 70:
            buy_size = 0
            sell_size += 15
            ask_price = best_bid + 1
        elif pos < -40:
            buy_size += 10
            sell_size = min(sell_size, 3)
            bid_price = min(best_ask - 1, max(bid_price, best_bid))

        pos = self._add_buy(orders, product, bid_price, buy_size, pos, limit)
        pos = self._add_sell(orders, product, ask_price, sell_size, pos, limit)

        data["last_mid"][product] = mid
        return orders

    # -------------------------------------------------------------------------
    # ASH_COATED_OSMIUM
    # -------------------------------------------------------------------------

    def trade_osmium(self, state: TradingState, order_depth: OrderDepth, data: Dict) -> List[Order]:
        product = self.OSMIUM
        limit = self.LIMITS[product]
        pos = self._position(state, product)

        mid = self._mid(order_depth)
        micro = self._microprice(order_depth)
        spread = self._spread(order_depth)
        imbalance = self._l1_imbalance(order_depth)

        if mid is None:
            return []

        short_ema = self._update_ema(data.get("osmium_short_ema"), mid, alpha=0.12)
        long_ema = self._update_ema(data.get("osmium_long_ema"), mid, alpha=0.025)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        # 10000 anchor + persistent wave tracker + microstructure tilt + inventory penalty
        wave_component = 0.70 * (short_ema - long_ema)
        micro_tilt = 0.90 * (micro - mid) if micro is not None else 0.0
        imbalance_tilt = 1.80 * imbalance
        anchor_pull = 0.12 * (10000.0 - mid)
        inventory_tilt = -0.12 * pos

        fair_value = 10000.0 + wave_component + micro_tilt + imbalance_tilt + anchor_pull + inventory_tilt

        orders: List[Order] = []

        # Wider spread -> more willing to capture immediate edge.
        wide_regime = spread is not None and spread >= 22
        narrow_regime = spread is not None and spread <= 17

        take_buy_edge = 1.2 if wide_regime else 1.8
        take_sell_edge = 1.2 if wide_regime else 1.8
        take_size = 20 if wide_regime else 10

        take_orders, pos = self._take_crossed_orders(
            product=product,
            order_depth=order_depth,
            fair_value=fair_value,
            pos=pos,
            limit=limit,
            buy_edge=take_buy_edge,
            sell_edge=take_sell_edge,
            max_take_per_side=take_size,
        )
        orders.extend(take_orders)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        signal = fair_value - mid
        abs_signal = abs(signal)

        if abs_signal >= 2.2:
            regime = "strong"
        elif abs_signal >= 0.9:
            regime = "medium"
        else:
            regime = "neutral"

        # For osmium, more symmetric than pepper, but still side-dominant under signal.
        if wide_regime:
            base_halfwidth = 2
            size_base = 14
        elif narrow_regime:
            base_halfwidth = 1
            size_base = 9
        else:
            base_halfwidth = 1
            size_base = 11

        # Improve by a tick on favored side when signal warrants it.
        improve_bid = 1 if (signal > 1.0 and spread >= 3) else 0
        improve_ask = 1 if (signal < -1.0 and spread >= 3) else 0

        bid_price = self._round_to_tick(fair_value - base_halfwidth)
        ask_price = self._round_to_tick(fair_value + base_halfwidth)

        bid_price = min(best_bid + improve_bid, bid_price)
        ask_price = max(best_ask - improve_ask, ask_price)

        # Do not cross with passive quotes
        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

        buy_size = size_base
        sell_size = size_base

        if regime == "strong":
            if signal > 0:
                buy_size += 9
                sell_size = max(4, sell_size - 5)
            else:
                sell_size += 9
                buy_size = max(4, buy_size - 5)
        elif regime == "medium":
            if signal > 0:
                buy_size += 5
                sell_size = max(5, sell_size - 2)
            else:
                sell_size += 5
                buy_size = max(5, buy_size - 2)

        # Inventory control
        if pos > 55:
            buy_size = min(buy_size, 2)
            sell_size += 10
            ask_price = max(best_bid + 1, min(ask_price, best_ask))
        elif pos > 70:
            buy_size = 0
            sell_size += 15
            ask_price = best_bid + 1
        elif pos < -55:
            sell_size = min(sell_size, 2)
            buy_size += 10
            bid_price = min(best_ask - 1, max(bid_price, best_bid))
        elif pos < -70:
            sell_size = 0
            buy_size += 15
            bid_price = best_ask - 1

        pos = self._add_buy(orders, product, bid_price, buy_size, pos, limit)
        pos = self._add_sell(orders, product, ask_price, sell_size, pos, limit)

        data["last_mid"][product] = mid
        return orders
