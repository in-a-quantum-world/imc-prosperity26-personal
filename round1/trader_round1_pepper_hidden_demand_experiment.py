from datamodel import OrderDepth, TradingState, Order
import jsonpickle

class Trader:
    PEPPER = "INTARIAN_PEPPER_ROOT"
    LIMIT = 80

    def run(self, state: TradingState):
        data = self._load_data(state)
        self._handle_day_reset(state, data)
        result = {self.PEPPER: []}
        if self.PEPPER in state.order_depths:
            result[self.PEPPER] = self.trade_pepper(state, state.order_depths[self.PEPPER], data)
        return result, 0, jsonpickle.encode(data)

    def _load_data(self, state):
        if getattr(state, "traderData", None):
            try:
                d = jsonpickle.decode(state.traderData)
                if isinstance(d, dict):
                    d.setdefault("last_timestamp", None)
                    d.setdefault("pepper_anchor", None)
                    d.setdefault("pepper_resid_ema", 0.0)
                    return d
            except:
                pass
        return {"last_timestamp": None, "pepper_anchor": None, "pepper_resid_ema": 0.0}

    def _handle_day_reset(self, state, data):
        last_ts = data.get("last_timestamp")
        if last_ts is not None and state.timestamp < last_ts:
            data["pepper_anchor"] = None
            data["pepper_resid_ema"] = 0.0
        data["last_timestamp"] = state.timestamp

    def _position(self, state):
        return state.position.get(self.PEPPER, 0)

    def _sorted_book(self, od):
        return sorted(od.buy_orders.items(), reverse=True), sorted(od.sell_orders.items())

    def _best_bid_ask(self, od):
        bids, asks = self._sorted_book(od)
        bb = bids[0][0] if bids else None
        ba = asks[0][0] if asks else None
        bv = bids[0][1] if bids else 0
        av = -asks[0][1] if asks else 0
        return bb, ba, bv, av

    def _mid(self, od):
        bb, ba, _, _ = self._best_bid_ask(od)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return float(bb)
        if ba is not None:
            return float(ba)
        return None

    def _microprice(self, od):
        bb, ba, bv, av = self._best_bid_ask(od)
        if bb is None or ba is None:
            return self._mid(od)
        t = bv + av
        return (ba * bv + bb * av) / t if t > 0 else (bb + ba) / 2.0

    def _spread(self, od):
        bb, ba, _, _ = self._best_bid_ask(od)
        return None if bb is None or ba is None else ba - bb

    def _imbalance(self, od):
        _, _, bv, av = self._best_bid_ask(od)
        d = bv + av
        return 0.0 if d <= 0 else (bv - av) / d

    def _ema(self, prev, x, alpha):
        return alpha * x + (1.0 - alpha) * prev

    def _add_buy(self, orders, price, qty, pos):
        qty = min(max(qty, 0), self.LIMIT - pos)
        if qty > 0:
            orders.append(Order(self.PEPPER, int(price), int(qty)))
            pos += qty
        return pos

    def _add_sell(self, orders, price, qty, pos):
        qty = min(max(qty, 0), self.LIMIT + pos)
        if qty > 0:
            orders.append(Order(self.PEPPER, int(price), int(-qty)))
            pos -= qty
        return pos

    def _target_position(self, t):
        if t < 5000:
            return 10
        if t < 15000:
            return 25
        if t < 35000:
            return 45
        if t < 80000:
            return 65
        if t < 980000:
            return 80
        return 50

    def _fair_value(self, state, od, data):
        mid = self._mid(od)
        micro = self._microprice(od)
        imb = self._imbalance(od)
        if mid is None:
            return 0.0

        raw_anchor = mid - 0.001 * state.timestamp
        snapped_anchor = 1000 * round(raw_anchor / 1000.0)

        prev_anchor = data.get("pepper_anchor")
        if prev_anchor is None:
            anchor = snapped_anchor
        else:
            anchor = snapped_anchor if abs(snapped_anchor - prev_anchor) >= 700 else prev_anchor
        data["pepper_anchor"] = anchor

        base_fv = anchor + 0.001 * state.timestamp
        residual = mid - base_fv
        resid_ema = self._ema(float(data.get("pepper_resid_ema", 0.0)), residual, 0.20)
        resid_ema = max(-3.0, min(3.0, resid_ema))
        data["pepper_resid_ema"] = resid_ema

        fv = base_fv + resid_ema
        if micro is not None:
            fv += 0.20 * (micro - mid)
        fv += 0.60 * imb
        return fv

    def trade_pepper(self, state, od, data):
        pos = self._position(state)
        bb, ba, _, _ = self._best_bid_ask(od)
        mid = self._mid(od)
        sp = self._spread(od)
        if bb is None or ba is None or mid is None:
            return []

        fv = self._fair_value(state, od, data)
        target = self._target_position(state.timestamp)
        gap = target - pos

        orders = []
        bids, asks = self._sorted_book(od)

        if gap > 0:
            t = state.timestamp
            if t < 10000:
                cross_edge = 1.5
            elif t < 40000:
                cross_edge = 2.5
            elif t < 120000:
                cross_edge = 4.0
            elif t < 900000:
                cross_edge = 3.0
            else:
                cross_edge = 2.0
            if gap >= 40:
                cross_edge += 0.5

            rem = min(gap, self.LIMIT - pos)
            for px, qsgn in asks:
                q = -qsgn
                if rem <= 0:
                    break
                if px <= fv + cross_edge:
                    take = min(q, rem)
                    pos = self._add_buy(orders, px, take, pos)
                    rem -= take
                else:
                    break

        gap = target - pos
        if sp is None:
            sp = 2
        bid_sizes = [20, 12, 7] if gap > 35 else [12, 8, 4] if gap > 10 else [5, 3, 2]
        bid_levels = [
            min(bb + (1 if sp >= 2 else 0), int(round(fv))),
            int(round(fv - 1)),
            int(round(fv - 2)),
        ]
        for px, qty in zip(bid_levels, bid_sizes):
            if px < ba:
                pos = self._add_buy(orders, px, qty, pos)

        # hidden-demand ask ladder
        if pos >= 50:
            if state.timestamp < 100000:
                ask_offsets = [7, 10, 14]
            elif state.timestamp < 900000:
                ask_offsets = [6, 9, 12]
            else:
                ask_offsets = [4, 7, 10]
            ask_sizes = [3, 2, 2] if pos >= 75 else [2, 1, 1]
            ask_levels = [max(ba, int(round(fv + off))) for off in ask_offsets]
            for px, qty in zip(ask_levels, ask_sizes):
                px = max(px, bb + 1)
                pos = self._add_sell(orders, px, qty, pos)

        rich = bb - fv
        if pos > target + 15 or rich >= 5 or state.timestamp > 995000:
            extra_asks = [max(bb + 1, int(round(fv + 2))), max(bb + 1, int(round(fv + 4)))]
            extra_sizes = [6, 4] if pos < 75 else [10, 6]
            for px, qty in zip(extra_asks, extra_sizes):
                pos = self._add_sell(orders, px, qty, pos)

        return orders
