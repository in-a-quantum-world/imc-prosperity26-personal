"""
Round 1 Trader v14 - MAXIMUM AGGRESSION push.

User reported v13 got 9.2k website on real platform. Big gap from my backtester's
~25k estimate => real passive fills are sparse.

CHANGES from v13:
  PEPPER
    1. NO early unwind. Hold +80 to t=999_500. Intel says unrealized is
       settled at fair value, so dumping early at the bid loses money.
       Only flatten in the LAST 500 ticks if we even need to.
    2. Even more aggressive entry (cross_edge 12 throughout the first
       300k ticks, then 8). Pay any price to get long fast.

  OSMIUM
    1. MUCH bigger maker sizes (20 base). If passive fills are rare,
       quote big to soak up whatever flow does come.
    2. LOWER take threshold (2.0 instead of 3.0) - take more often.
    3. CROSSING limit on takes - extend take_size to 40 to walk the book
       harder when signal is strong.
    4. Tighter quotes on the side that fights inventory.

If v14 still hits the same wall, the bottleneck is structural, not strategic:
either round 1 evaluates 1 day (capping you near 9k website) or there's a
cheaper edge we haven't found (puzzle / hidden alpha).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle


PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"


class Trader:

    LIMITS = {PEPPER: 80, OSMIUM: 80}

    PEPPER_DRIFT = 0.001
    PEPPER_ANCHOR_HIST_LEN = 50
    # HOLD ALL THE WAY. Only flatten in last 500 ticks.
    PEPPER_FLATTEN_FROM = 999_500

    OSMIUM_BASE_SIZE = 20
    OSMIUM_TAKE_THRESHOLD = 2.0
    OSMIUM_TAKE_EDGE = 1.0
    OSMIUM_MAX_TAKE = 40
    OSMIUM_ANCHOR_BLEND = 0.5
    OSMIUM_FV_HIST_LEN = 80

    # ------------------------------------------------------------------

    def run(self, state: TradingState):
        data = self._load_data(state)
        self._handle_day_reset(state, data)

        result: Dict[str, List[Order]] = {PEPPER: [], OSMIUM: []}

        if PEPPER in state.order_depths:
            result[PEPPER] = self.trade_pepper(state, state.order_depths[PEPPER], data)

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self.trade_osmium(state, state.order_depths[OSMIUM], data)

        return result, 0, jsonpickle.encode(data)

    # state ------------------------------------------------------------

    def _load_data(self, state: TradingState):
        if getattr(state, "traderData", None):
            try:
                d = jsonpickle.decode(state.traderData)
                if isinstance(d, dict):
                    d.setdefault("last_ts", None)
                    d.setdefault("pepper_anchor_samples", [])
                    d.setdefault("osmium_short_ema", None)
                    d.setdefault("osmium_long_ema", None)
                    d.setdefault("osmium_last_imbalance", 0.0)
                    d.setdefault("osmium_mid_hist", [])
                    d.setdefault("osmium_wallmid_hist", [])
                    return d
            except Exception:
                pass
        return {
            "last_ts": None,
            "pepper_anchor_samples": [],
            "osmium_short_ema": None,
            "osmium_long_ema": None,
            "osmium_last_imbalance": 0.0,
            "osmium_mid_hist": [],
            "osmium_wallmid_hist": [],
        }

    def _handle_day_reset(self, state: TradingState, data):
        last_ts = data.get("last_ts")
        if last_ts is not None and state.timestamp < last_ts:
            data["pepper_anchor_samples"] = []
            data["osmium_short_ema"] = None
            data["osmium_long_ema"] = None
            data["osmium_last_imbalance"] = 0.0
            data["osmium_mid_hist"] = []
            data["osmium_wallmid_hist"] = []
        data["last_ts"] = state.timestamp

    # helpers ----------------------------------------------------------

    @staticmethod
    def _best(od):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    @staticmethod
    def _mid(od):
        bb, ba = Trader._best(od)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    @staticmethod
    def _wall_levels(od):
        wb = wa = None; wb_size = wa_size = 0
        if od.buy_orders:
            wb_p, wb_s = max(od.buy_orders.items(), key=lambda kv: kv[1])
            wb, wb_size = wb_p, wb_s
        if od.sell_orders:
            wa_p, wa_s_neg = max(od.sell_orders.items(), key=lambda kv: -kv[1])
            wa, wa_size = wa_p, -wa_s_neg
        return wb, wa, wb_size, wa_size

    @staticmethod
    def _wall_mid(od):
        wb, wa, _, _ = Trader._wall_levels(od)
        return (wb + wa) / 2.0 if (wb is not None and wa is not None) else Trader._mid(od)

    @staticmethod
    def _imbalance(od):
        if not od.buy_orders or not od.sell_orders: return 0.0
        bv = od.buy_orders[max(od.buy_orders)]
        av = -od.sell_orders[min(od.sell_orders)]
        d = bv + av
        return 0.0 if d <= 0 else (bv - av) / d

    @staticmethod
    def _ema(prev, x, alpha): return x if prev is None else alpha * x + (1.0 - alpha) * prev

    @staticmethod
    def _add_buy(orders, prod, price, qty, pos, limit):
        if qty <= 0: return pos
        qty = min(qty, limit - pos)
        if qty > 0:
            orders.append(Order(prod, int(price), int(qty)))
            pos += qty
        return pos

    @staticmethod
    def _add_sell(orders, prod, price, qty, pos, limit):
        if qty <= 0: return pos
        qty = min(qty, limit + pos)
        if qty > 0:
            orders.append(Order(prod, int(price), -int(qty)))
            pos -= qty
        return pos

    # PEPPER -----------------------------------------------------------

    def _pepper_anchor(self, state, od, data):
        wm = self._wall_mid(od)
        if wm is None: return None
        sample = wm - self.PEPPER_DRIFT * state.timestamp
        s = data["pepper_anchor_samples"]
        s.append(sample)
        if len(s) > self.PEPPER_ANCHOR_HIST_LEN: del s[0]
        ss = sorted(s)
        m = ss[len(ss) // 2]
        return 1000.0 * round(m / 1000.0)

    def trade_pepper(self, state, od, data):
        pos = state.position.get(PEPPER, 0)
        limit = self.LIMITS[PEPPER]
        bb, ba = self._best(od)
        if bb is None or ba is None: return []

        anchor = self._pepper_anchor(state, od, data)
        if anchor is None: return []
        fv = anchor + self.PEPPER_DRIFT * state.timestamp
        t = state.timestamp
        orders = []

        # ONLY flatten in the very last 500 ticks if positions exist.
        # If unrealized is settled at FV, this is mostly a safety net.
        if t >= self.PEPPER_FLATTEN_FROM and pos > 0:
            # walk the bid book taking everything available
            for px, qty in sorted(od.buy_orders.items(), reverse=True):
                if pos <= 0: break
                take = min(qty, pos)
                pos = self._add_sell(orders, PEPPER, px, take, pos, limit)
            return orders

        # If we're already at +80, just rest a bid for cheap restocks
        if pos >= limit:
            bid_price = bb + 1 if (ba - bb) >= 2 else bb
            pos = self._add_buy(orders, PEPPER, bid_price, 4, pos, limit)
            return orders

        # MAX-AGGRESSION ENTRY -- pay anything reasonable to get long fast.
        # Each tick at -1 share costs 0.001 in mtm; we pay this back fast.
        if t < 300_000:
            cross_edge = 12.0
        elif t < 700_000:
            cross_edge = 8.0
        else:
            cross_edge = 5.0

        target = limit
        gap = target - pos

        # Walk asks within edge
        for px, neg_qty in sorted(od.sell_orders.items()):
            qty = -neg_qty
            if px <= fv + cross_edge:
                take = min(qty, gap)
                if take > 0:
                    pos = self._add_buy(orders, PEPPER, px, take, pos, limit)
                    gap = target - pos
                    if gap <= 0: break
            else:
                break

        # Rest a passive bid below
        if pos < limit:
            spread = ba - bb
            bid_price = min(bb + 1, int(round(fv + 1))) if spread >= 2 else bb
            if bid_price >= ba: bid_price = ba - 1
            qty = min(limit - pos, max(15, gap // 2))
            pos = self._add_buy(orders, PEPPER, bid_price, qty, pos, limit)

        return orders

    # OSMIUM -----------------------------------------------------------

    def trade_osmium(self, state, od, data):
        pos = state.position.get(OSMIUM, 0)
        limit = self.LIMITS[OSMIUM]

        bb, ba = self._best(od)
        if bb is None or ba is None: return []

        wb, wa, wb_sz, wa_sz = self._wall_levels(od)
        if wb is None or wa is None: return []

        mid = (bb + ba) / 2.0
        spread = ba - bb
        imb = self._imbalance(od)

        wall_mid = (wb + wa) / 2.0 if (wb_sz >= 5 and wa_sz >= 5) else mid

        wmh = data["osmium_wallmid_hist"]
        wmh.append(wall_mid)
        if len(wmh) > self.OSMIUM_FV_HIST_LEN: del wmh[0]
        wm_smooth = sum(wmh[-30:]) / min(30, len(wmh))

        anchor = (self.OSMIUM_ANCHOR_BLEND * wm_smooth
                  + (1 - self.OSMIUM_ANCHOR_BLEND) * 10000.0)

        short_ema = self._ema(data.get("osmium_short_ema"), mid, 0.10)
        long_ema = self._ema(data.get("osmium_long_ema"), mid, 0.03)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        mh = data["osmium_mid_hist"]
        mh.append(mid)
        if len(mh) > 60: del mh[0]
        recent = mh[-30:] if len(mh) >= 10 else mh
        n = len(recent)
        mu = sum(recent) / n if n else mid
        var = sum((x - mu) ** 2 for x in recent) / max(1, n)
        sigma = max(var ** 0.5, 1e-6)
        z = (mid - mu) / sigma

        last_imb = data.get("osmium_last_imbalance", 0.0)
        imb_change = imb - last_imb
        data["osmium_last_imbalance"] = imb

        fv = anchor
        fv += 0.70 * (short_ema - long_ema)
        fv += 0.80 * imb
        fv += 0.05 * (anchor - mid)
        fv += -0.55 * z
        fv += -0.05 * pos
        fv += 0.15 * imb_change

        # Halfwidth & sizing
        if spread >= 24:
            halfwidth = 4; base_size = self.OSMIUM_BASE_SIZE + 4
        elif spread >= 18:
            halfwidth = 3; base_size = self.OSMIUM_BASE_SIZE + 2
        else:
            halfwidth = 2; base_size = self.OSMIUM_BASE_SIZE

        if abs(z) > 2.5:
            base_size = max(8, base_size - 4)

        orders = []
        signal = fv - mid

        # ---- AGGRESSIVE TAKES (lower threshold, larger size) -----------
        if signal > self.OSMIUM_TAKE_THRESHOLD and pos < limit:
            for px, neg_qty in sorted(od.sell_orders.items()):
                qty = -neg_qty
                if px <= fv - self.OSMIUM_TAKE_EDGE:
                    take = min(qty, self.OSMIUM_MAX_TAKE, limit - pos)
                    if take > 0:
                        pos = self._add_buy(orders, OSMIUM, px, take, pos, limit)
                else:
                    break

        if signal < -self.OSMIUM_TAKE_THRESHOLD and pos > -limit:
            for px, qty in sorted(od.buy_orders.items(), reverse=True):
                if px >= fv + self.OSMIUM_TAKE_EDGE:
                    take = min(qty, self.OSMIUM_MAX_TAKE, limit + pos)
                    if take > 0:
                        pos = self._add_sell(orders, OSMIUM, px, take, pos, limit)
                else:
                    break

        # ---- MAKER QUOTES ---------------------------------------------
        improve_bid = 1 if signal > 0.5 and spread >= 2 else 0
        improve_ask = 1 if signal < -0.5 and spread >= 2 else 0

        bid_price = min(bb + improve_bid, int(round(fv - halfwidth)))
        ask_price = max(ba - improve_ask, int(round(fv + halfwidth)))

        # Force inside the wall when wall is healthy
        if wb_sz >= 10 and bid_price <= wb:
            bid_price = wb + 1
        if wa_sz >= 10 and ask_price >= wa:
            ask_price = wa - 1

        if bid_price >= ba: bid_price = ba - 1
        if ask_price <= bb: ask_price = bb + 1
        if ask_price - bid_price < 2:
            mq = (bid_price + ask_price) // 2
            bid_price = mq - 1; ask_price = mq + 1

        buy_size = base_size
        sell_size = base_size

        if signal > 0.5:
            buy_size += 8; sell_size = max(3, sell_size - 4)
        elif signal < -0.5:
            sell_size += 8; buy_size = max(3, buy_size - 4)

        if pos > 50:
            buy_size = max(0, buy_size - 12); sell_size += 8
        elif pos > 25:
            buy_size = max(2, buy_size - 6); sell_size += 4
        elif pos < -50:
            sell_size = max(0, sell_size - 12); buy_size += 8
        elif pos < -25:
            sell_size = max(2, sell_size - 6); buy_size += 4

        buy_size = min(buy_size, max(0, limit - pos))
        sell_size = min(sell_size, max(0, limit + pos))

        pos = self._add_buy(orders, OSMIUM, bid_price, buy_size, pos, limit)
        pos = self._add_sell(orders, OSMIUM, ask_price, sell_size, pos, limit)

        return orders