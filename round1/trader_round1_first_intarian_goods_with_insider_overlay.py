
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Tuple, Optional
import jsonpickle


class Trader:
    """
    Round 1 trader for:
      - INTARIAN_PEPPER_ROOT
      - ASH_COATED_OSMIUM

    Architecture:
      1) Base market-making engine remains:
         - PEPPER: deterministic intraday drift + microstructure tilt
         - OSMIUM: 10000 anchor + persistence tracker + microstructure tilt

      2) Insider / informed-flow overlay added:
         - tracks running daily trade extremes (min/max)
         - watches for trades at those extremes in the expected direction
         - boosts confidence with larger trade size
         - decays signals over time
         - cancels signals on contradicting new extrema
         - uses active bias to adjust fair value, quote skew, sizes, and take thresholds

    Intended behavior:
      - PEPPER: mainly follow informed dip-buying at rolling lows
      - OSMIUM: use both long-at-low and short-at-high overlays
    """

    LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    PEPPER = "INTARIAN_PEPPER_ROOT"
    OSMIUM = "ASH_COATED_OSMIUM"

    def run(self, state: TradingState):
        data = self._load_data(state)

        # Reset per-day state if timestamp rolled back
        self._handle_day_reset(state, data)

        # Update insider tracker using market trades before quoting
        self._update_insider_signals(state, data)

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

    def _blank_tracker(self) -> Dict:
        return {
            "running_min_trade": None,
            "running_max_trade": None,
            "bias": 0,                  # -1 short, 0 none, +1 long
            "strength": 0.0,            # confidence level
            "last_signal_ts": None,
            "last_update_ts": None,
        }

    def _load_data(self, state: TradingState) -> Dict:
        if getattr(state, "traderData", None):
            try:
                data = jsonpickle.decode(state.traderData)
                if isinstance(data, dict):
                    # Backfill missing keys
                    if "pepper_anchor" not in data:
                        data["pepper_anchor"] = None
                    if "osmium_short_ema" not in data:
                        data["osmium_short_ema"] = None
                    if "osmium_long_ema" not in data:
                        data["osmium_long_ema"] = None
                    if "last_mid" not in data:
                        data["last_mid"] = {self.PEPPER: None, self.OSMIUM: None}
                    if "last_timestamp" not in data:
                        data["last_timestamp"] = None
                    if "insider" not in data:
                        data["insider"] = {
                            self.PEPPER: self._blank_tracker(),
                            self.OSMIUM: self._blank_tracker(),
                        }
                    else:
                        for product in [self.PEPPER, self.OSMIUM]:
                            if product not in data["insider"]:
                                data["insider"][product] = self._blank_tracker()
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
            "last_timestamp": None,
            "insider": {
                self.PEPPER: self._blank_tracker(),
                self.OSMIUM: self._blank_tracker(),
            },
        }

    def _handle_day_reset(self, state: TradingState, data: Dict) -> None:
        last_ts = data.get("last_timestamp")
        if last_ts is not None and state.timestamp < last_ts:
            # New day in Prosperity backtests usually resets timestamp.
            data["pepper_anchor"] = None
            data["osmium_short_ema"] = None
            data["osmium_long_ema"] = None
            data["insider"] = {
                self.PEPPER: self._blank_tracker(),
                self.OSMIUM: self._blank_tracker(),
            }
        data["last_timestamp"] = state.timestamp

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
    # Insider tracker
    # -------------------------------------------------------------------------

    def _all_market_trades(self, state: TradingState, product: str) -> List[Trade]:
        trades = state.market_trades.get(product, [])
        # Sort by timestamp if available, otherwise leave as is
        try:
            return sorted(trades, key=lambda t: t.timestamp)
        except Exception:
            return trades

    def _trade_is_bullish(self, trade: Trade, mid: Optional[float]) -> bool:
        if mid is None:
            return False
        try:
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            if buyer == "SUBMISSION":
                return True
            if seller == "SUBMISSION":
                return False
        except Exception:
            pass
        return trade.price <= mid

    def _trade_is_bearish(self, trade: Trade, mid: Optional[float]) -> bool:
        if mid is None:
            return False
        try:
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            if seller == "SUBMISSION":
                return True
            if buyer == "SUBMISSION":
                return False
        except Exception:
            pass
        return trade.price >= mid

    def _decay_insider(self, tracker: Dict, state_ts: int, decay_per_100: float = 0.15) -> None:
        last_signal_ts = tracker.get("last_signal_ts")
        if last_signal_ts is None:
            return
        dt = max(0, state_ts - last_signal_ts)
        decay = decay_per_100 * (dt / 100.0)
        tracker["strength"] = max(0.0, tracker["strength"] - decay)
        if tracker["strength"] <= 0.15:
            tracker["bias"] = 0
            tracker["strength"] = 0.0

    def _score_trade_signal(
        self,
        product: str,
        trade: Trade,
        mid: Optional[float],
        at_low: bool,
        at_high: bool,
    ) -> Tuple[float, float]:
        """
        Returns (long_score, short_score)
        """
        size = abs(trade.quantity)
        long_score = 0.0
        short_score = 0.0

        if at_low:
            long_score += 1.0
            if self._trade_is_bullish(trade, mid):
                long_score += 1.0

            # size bonus
            if product == self.OSMIUM:
                if size >= 17:
                    long_score += 1.0
                elif size >= 12:
                    long_score += 0.5
            else:  # PEPPER
                if size >= 15:
                    long_score += 1.0
                elif size >= 10:
                    long_score += 0.5

        if at_high:
            short_score += 1.0
            if self._trade_is_bearish(trade, mid):
                short_score += 1.0

            if product == self.OSMIUM:
                if size >= 17:
                    short_score += 1.0
                elif size >= 12:
                    short_score += 0.5
            else:
                # Pepper short signals matter much less; keep weak
                if size >= 18:
                    short_score += 0.6
                elif size >= 12:
                    short_score += 0.3

        return long_score, short_score

    def _update_tracker_for_product(self, state: TradingState, product: str, data: Dict) -> None:
        tracker = data["insider"][product]
        trades = self._all_market_trades(state, product)
        order_depth = state.order_depths.get(product)
        mid = self._mid(order_depth) if order_depth is not None else None

        # Decay old signal each run
        self._decay_insider(tracker, state.timestamp, decay_per_100=0.18 if product == self.OSMIUM else 0.12)

        for trade in trades:
            price = trade.price

            # Update running extrema
            if tracker["running_min_trade"] is None or price < tracker["running_min_trade"]:
                tracker["running_min_trade"] = price
            if tracker["running_max_trade"] is None or price > tracker["running_max_trade"]:
                tracker["running_max_trade"] = price

            at_low = tracker["running_min_trade"] is not None and price <= tracker["running_min_trade"] + 1
            at_high = tracker["running_max_trade"] is not None and price >= tracker["running_max_trade"] - 1

            long_score, short_score = self._score_trade_signal(product, trade, mid, at_low, at_high)

            # Contradiction logic first
            if tracker["bias"] > 0 and at_low and self._trade_is_bearish(trade, mid):
                tracker["strength"] *= 0.6
            if tracker["bias"] > 0 and at_high and short_score >= 2.0:
                tracker["strength"] *= 0.4
                if product == self.OSMIUM:
                    tracker["bias"] = -1
                    tracker["strength"] = max(tracker["strength"], short_score - 0.5)
                    tracker["last_signal_ts"] = state.timestamp

            if tracker["bias"] < 0 and at_high and self._trade_is_bullish(trade, mid):
                tracker["strength"] *= 0.6
            if tracker["bias"] < 0 and at_low and long_score >= 2.0:
                tracker["strength"] *= 0.4
                tracker["bias"] = 1
                tracker["strength"] = max(tracker["strength"], long_score - 0.5)
                tracker["last_signal_ts"] = state.timestamp

            # Activation logic
            if long_score >= 2.0:
                # Pepper: primarily long-following
                tracker["bias"] = 1
                tracker["strength"] = max(tracker["strength"], long_score)
                tracker["last_signal_ts"] = state.timestamp

            if short_score >= 2.0:
                # Osmium uses both sides, Pepper only weakly
                if product == self.OSMIUM:
                    tracker["bias"] = -1
                    tracker["strength"] = max(tracker["strength"], short_score)
                    tracker["last_signal_ts"] = state.timestamp
                else:
                    # Pepper: allow only a mild weakening, not full short conviction
                    tracker["strength"] = min(tracker["strength"], 1.5)

        # Timeout expiry
        last_signal_ts = tracker.get("last_signal_ts")
        if last_signal_ts is not None:
            ttl = 220 if product == self.PEPPER else 160
            if state.timestamp - last_signal_ts > ttl and tracker["strength"] < 1.5:
                tracker["bias"] = 0
                tracker["strength"] = 0.0

        tracker["last_update_ts"] = state.timestamp

    def _update_insider_signals(self, state: TradingState, data: Dict) -> None:
        for product in [self.PEPPER, self.OSMIUM]:
            if product in state.order_depths:
                self._update_tracker_for_product(state, product, data)

    def _insider_adjustment(self, product: str, data: Dict) -> Tuple[float, float, int]:
        """
        Returns:
          fv_shift, aggression_bonus, bias
        """
        tracker = data["insider"][product]
        bias = tracker["bias"]
        strength = tracker["strength"]

        if bias == 0 or strength <= 0:
            return 0.0, 0.0, 0

        if product == self.PEPPER:
            # Pepper: mainly a long-at-lows amplifier
            if bias > 0:
                fv_shift = min(3.0, 0.9 * strength)
                aggr = min(1.6, 0.45 * strength)
                return fv_shift, aggr, 1
            return 0.0, 0.0, 0

        # Osmium can use both long and short overlays
        fv_shift = min(2.5, 0.75 * strength) * (1 if bias > 0 else -1)
        aggr = min(1.4, 0.40 * strength)
        return fv_shift, aggr, bias

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
        if mid is None:
            return []

        raw_anchor = mid - 0.001 * state.timestamp
        snapped_anchor = 1000 * round(raw_anchor / 1000.0)

        prev_anchor = data.get("pepper_anchor")
        if prev_anchor is None:
            anchor = snapped_anchor
        else:
            if abs(snapped_anchor - prev_anchor) >= 700:
                anchor = snapped_anchor
            else:
                anchor = 0.90 * prev_anchor + 0.10 * snapped_anchor
        data["pepper_anchor"] = anchor

        drift_fv = anchor + 0.001 * state.timestamp
        micro_tilt = 0.80 * (micro - mid) if micro is not None else 0.0
        imbalance_tilt = 2.50 * self._l1_imbalance(order_depth)
        inventory_tilt = -0.10 * pos

        insider_shift, insider_aggr, insider_bias = self._insider_adjustment(product, data)

        fair_value = drift_fv + micro_tilt + imbalance_tilt + inventory_tilt + insider_shift

        orders: List[Order] = []

        # With insider long active, slightly lower the threshold to lift asks.
        take_orders, pos = self._take_crossed_orders(
            product=product,
            order_depth=order_depth,
            fair_value=fair_value,
            pos=pos,
            limit=limit,
            buy_edge=max(0.8, 2.0 - insider_aggr),
            sell_edge=2.3,
            max_take_per_side=22 if insider_bias > 0 else 18,
        )
        orders.extend(take_orders)

        best_bid, best_ask, _, _ = self._best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        signal = fair_value - mid
        abs_signal = abs(signal)

        if abs_signal >= 3.0:
            regime = "strong"
        elif abs_signal >= 1.2:
            regime = "medium"
        else:
            regime = "neutral"

        if spread >= 12:
            buy_offset = 1
            sell_offset = 2
        else:
            buy_offset = 0
            sell_offset = 1

        long_pressure = pos / limit

        buy_bias = 0.0
        if long_pressure < 0.50:
            buy_bias += 0.8
        if long_pressure < 0.20:
            buy_bias += 0.6
        if signal > 0:
            buy_bias += 0.7
        if self._l1_imbalance(order_depth) > 0.15:
            buy_bias += 0.5
        if insider_bias > 0:
            buy_bias += 0.8 + 0.4 * insider_aggr

        sell_bias = 0.0
        if long_pressure > 0.60:
            sell_bias += 1.0
        if long_pressure > 0.80:
            sell_bias += 0.8
        if signal < 0:
            sell_bias += 0.5

        improve_bid = 1 if ((regime != "neutral" or insider_bias > 0) and spread >= 3) else 0
        improve_ask = 1 if (regime == "strong" and signal < -2 and spread >= 3) else 0

        bid_price = min(
            best_bid + improve_bid,
            self._round_to_tick(fair_value - buy_offset + buy_bias * 0.3)
        )
        ask_price = max(
            best_ask - improve_ask,
            self._round_to_tick(fair_value + sell_offset + sell_bias * 0.3)
        )

        if bid_price >= best_ask:
            bid_price = best_ask - 1
        if ask_price <= best_bid:
            ask_price = best_bid + 1

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

        if insider_bias > 0:
            buy_size += 6 + int(3 * insider_aggr)
            sell_size = max(2, sell_size - 4)

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
        if mid is None:
            return []

        short_ema = self._update_ema(data.get("osmium_short_ema"), mid, alpha=0.12)
        long_ema = self._update_ema(data.get("osmium_long_ema"), mid, alpha=0.025)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        wave_component = 0.70 * (short_ema - long_ema)
        micro_tilt = 0.90 * (micro - mid) if micro is not None else 0.0
        imbalance_tilt = 1.80 * self._l1_imbalance(order_depth)
        anchor_pull = 0.12 * (10000.0 - mid)
        inventory_tilt = -0.12 * pos

        insider_shift, insider_aggr, insider_bias = self._insider_adjustment(product, data)

        fair_value = 10000.0 + wave_component + micro_tilt + imbalance_tilt + anchor_pull + inventory_tilt + insider_shift

        orders: List[Order] = []

        wide_regime = spread is not None and spread >= 22
        narrow_regime = spread is not None and spread <= 17

        take_buy_edge = (1.2 if wide_regime else 1.8) - (0.5 * insider_aggr if insider_bias > 0 else 0.0)
        take_sell_edge = (1.2 if wide_regime else 1.8) - (0.5 * insider_aggr if insider_bias < 0 else 0.0)
        take_size = 20 if wide_regime else 10
        if insider_bias != 0:
            take_size += 6

        take_orders, pos = self._take_crossed_orders(
            product=product,
            order_depth=order_depth,
            fair_value=fair_value,
            pos=pos,
            limit=limit,
            buy_edge=max(0.8, take_buy_edge),
            sell_edge=max(0.8, take_sell_edge),
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

        if wide_regime:
            base_halfwidth = 2
            size_base = 14
        elif narrow_regime:
            base_halfwidth = 1
            size_base = 9
        else:
            base_halfwidth = 1
            size_base = 11

        improve_bid = 1 if ((signal > 1.0 or insider_bias > 0) and spread >= 3) else 0
        improve_ask = 1 if ((signal < -1.0 or insider_bias < 0) and spread >= 3) else 0

        bid_price = self._round_to_tick(fair_value - base_halfwidth)
        ask_price = self._round_to_tick(fair_value + base_halfwidth)

        bid_price = min(best_bid + improve_bid, bid_price)
        ask_price = max(best_ask - improve_ask, ask_price)

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

        if insider_bias > 0:
            buy_size += 5 + int(3 * insider_aggr)
            sell_size = max(3, sell_size - 4)
        elif insider_bias < 0:
            sell_size += 5 + int(3 * insider_aggr)
            buy_size = max(3, buy_size - 4)

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
