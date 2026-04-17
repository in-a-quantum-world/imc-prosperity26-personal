"""
Round 1 Trader v13 — final consolidated recommendation.

Backtester results (3 days of training data):
  v6 (baseline): Pepper 238k + Osmium 14k = 252k actual (~25.3k website-est)
  v13 (this):    Pepper 238k + Osmium ~16k = 254k actual (~25.4k website-est)
                 with bigger MM size, the 50%-capture upside is ~26.8k.

WHY THIS VERSION

  PEPPER  -- Empirically near theoretical max. The work here is bulletproofing,
             not chasing more PnL.
       1. ANCHOR INFERENCE FROM WALL_MID. The day-anchor (round number ending
          in 000) is recovered from a robust median of (wall_mid - drift*ts)
          over the last 50 ticks, then rounded to nearest 1000. Wall_mid has
          1/3 the variance of mid for this purpose, so the median converges
          much faster and survives noise.
       2. STATE PERSISTED VIA traderData. Per Jasper/jmerle's admin warning:
          do not rely on global object persistence. Everything is in the data
          dict that gets jsonpickled.
       3. AGGRESSIVE EARLY ENTRY (+10 cross_edge for first 50k ticks). Each
          tick of waiting costs 0.001 in mtm; paying +5 above fv to enter
          breaks even at 5k ticks. Early-aggressive is correct.
       4. CLEAN END-OF-DAY EXIT at the best ask (passive) — never market-sell
          into a thin bid.

  OSMIUM  -- Signal-based MM around an anchor. Kept v6's working signal mix
             because every "cleaner" rewrite I tested under-performed it.
       1. ANCHOR is mostly the smoothed wall_mid (live), with a small constant
          regularization toward 10000 so we don't drift if walls misbehave.
       2. SIGNAL same as v6: short-EMA momentum + book imbalance + mean-rev
          to anchor + z-score reversion + inventory penalty + imb-change.
          (Tested clean rewrites; this composite empirically wins.)
       3. QUOTE PLACEMENT FORCED INSIDE THE WALL when wall is healthy. The
          maker at wb+2/wa-2 (visible in trade clusters) is competition; we
          outbid them at wb+3 etc. when fv allows.
       4. LARGER MAKER SIZE (10-12 base). v6's size-5 leaves capacity on the
          table when quotes do get hit.
       5. TAKE ORDERS retained but slightly tighter threshold so we don't
          fade the anchor too eagerly.

KEY TUNABLES (search "TUNABLE" to find them)
  - PEPPER_HOLD_UNTIL / DUMP_FROM: when to start unwinding
  - PEPPER cross_edge schedule: how aggressive to enter
  - OSMIUM_BASE_SIZE: size of resting maker quotes
  - OSMIUM_TAKE_THRESHOLD: how strong the signal must be before crossing
  - OSMIUM_ANCHOR_BLEND: how much wall_mid vs constant-10000

A/B IDEAS FOR THE REAL ROUND
  A. Set OSMIUM_BASE_SIZE = 18 (very aggressive). If real fills are decent,
     this captures more spread.
  B. Set PEPPER_DUMP_FROM = 999_500 (later exit). If the unrealized-at-mid
     settlement is real, holding longer is +EV.
  C. Set OSMIUM_TAKE_THRESHOLD = 4.0 (more conservative takes). Reduces drift
     bias if v6's takes were lucky-directional in training.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import jsonpickle


PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"


class Trader:

    LIMITS = {PEPPER: 80, OSMIUM: 80}

    # ---------- PEPPER TUNABLES ----------
    PEPPER_DRIFT = 0.001
    PEPPER_ANCHOR_HIST_LEN = 50
    PEPPER_HOLD_UNTIL = 985_000
    PEPPER_DUMP_FROM = 998_000

    # ---------- OSMIUM TUNABLES ----------
    OSMIUM_HARD_LIMIT = 80
    OSMIUM_SOFT_LIMIT = 80
    OSMIUM_BASE_SIZE = 10
    OSMIUM_TAKE_THRESHOLD = 3.0
    OSMIUM_TAKE_EDGE = 1.5
    OSMIUM_ANCHOR_BLEND = 0.5     # 0=pure 10000, 1=pure wall_mid (smoothed)
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

        traderData = jsonpickle.encode(data)
        return result, 0, traderData

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # book helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best(od: OrderDepth):
        bb = max(od.buy_orders) if od.buy_orders else None
        ba = min(od.sell_orders) if od.sell_orders else None
        return bb, ba

    @staticmethod
    def _mid(od: OrderDepth):
        bb, ba = Trader._best(od)
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    @staticmethod
    def _wall_levels(od: OrderDepth):
        wb = wa = None
        wb_size = wa_size = 0
        if od.buy_orders:
            wb_p, wb_s = max(od.buy_orders.items(), key=lambda kv: kv[1])
            wb, wb_size = wb_p, wb_s
        if od.sell_orders:
            wa_p, wa_s_neg = max(od.sell_orders.items(), key=lambda kv: -kv[1])
            wa, wa_size = wa_p, -wa_s_neg
        return wb, wa, wb_size, wa_size

    @staticmethod
    def _wall_mid(od: OrderDepth):
        wb, wa, wb_sz, wa_sz = Trader._wall_levels(od)
        if wb is None or wa is None:
            return Trader._mid(od)
        return (wb + wa) / 2.0

    @staticmethod
    def _imbalance(od: OrderDepth):
        if not od.buy_orders or not od.sell_orders:
            return 0.0
        bv = od.buy_orders[max(od.buy_orders)]
        av = -od.sell_orders[min(od.sell_orders)]
        denom = bv + av
        return 0.0 if denom <= 0 else (bv - av) / denom

    @staticmethod
    def _spread(od: OrderDepth):
        bb, ba = Trader._best(od)
        if bb is None or ba is None:
            return None
        return ba - bb

    @staticmethod
    def _update_ema(prev, x, alpha):
        if prev is None:
            return x
        return alpha * x + (1.0 - alpha) * prev

    @staticmethod
    def _add_buy(orders, prod, price, qty, pos, limit):
        if qty <= 0:
            return pos
        cap = limit - pos
        qty = min(qty, cap)
        if qty > 0:
            orders.append(Order(prod, int(price), int(qty)))
            pos += qty
        return pos

    @staticmethod
    def _add_sell(orders, prod, price, qty, pos, limit):
        if qty <= 0:
            return pos
        cap = limit + pos
        qty = min(qty, cap)
        if qty > 0:
            orders.append(Order(prod, int(price), -int(qty)))
            pos -= qty
        return pos

    # ------------------------------------------------------------------
    # PEPPER
    # ------------------------------------------------------------------

    def _pepper_anchor(self, state, od, data):
        wm = self._wall_mid(od)
        if wm is None:
            return None
        sample = wm - self.PEPPER_DRIFT * state.timestamp
        samples = data["pepper_anchor_samples"]
        samples.append(sample)
        if len(samples) > self.PEPPER_ANCHOR_HIST_LEN:
            del samples[0]
        s = sorted(samples)
        m = s[len(s) // 2]
        return 1000.0 * round(m / 1000.0)

    def trade_pepper(self, state: TradingState, od: OrderDepth, data):
        pos = state.position.get(PEPPER, 0)
        limit = self.LIMITS[PEPPER]
        bb, ba = self._best(od)
        if bb is None or ba is None:
            return []

        anchor = self._pepper_anchor(state, od, data)
        if anchor is None:
            return []
        fv = anchor + self.PEPPER_DRIFT * state.timestamp
        t = state.timestamp
        orders: List[Order] = []

        # End-of-day flatten
        if t >= self.PEPPER_DUMP_FROM and pos > 0:
            pos = self._add_sell(orders, PEPPER, ba, pos, pos, limit)
            return orders

        # Begin gradual unwind
        if t >= self.PEPPER_HOLD_UNTIL and pos > 20:
            sell_price = max(ba - 1, int(round(fv + 1)))
            pos = self._add_sell(orders, PEPPER, sell_price, min(pos - 20, 15), pos, limit)
            return orders

        # Accumulate to +80
        target = limit
        gap = target - pos
        if gap <= 0:
            bid_price = bb + 1 if (ba - bb) >= 2 else bb
            pos = self._add_buy(orders, PEPPER, bid_price, 4, pos, limit)
            return orders

        # Time-decaying willingness to pay
        if t < 50_000:
            cross_edge = 10.0
        elif t < 200_000:
            cross_edge = 8.0
        elif t < 600_000:
            cross_edge = 6.0
        else:
            cross_edge = 4.0
        if gap >= 60:
            cross_edge += 2.0

        # Walk the ask side taking everything within edge
        for px, neg_qty in sorted(od.sell_orders.items()):
            qty = -neg_qty
            if px <= fv + cross_edge:
                take = min(qty, gap)
                if take > 0:
                    pos = self._add_buy(orders, PEPPER, px, take, pos, limit)
                    gap = target - pos
                    if gap <= 0:
                        break
            else:
                break

        # Rest a passive bid for cheap fills
        if pos < limit:
            spread = ba - bb
            bid_price = min(bb + 1, int(round(fv + 1))) if spread >= 2 else bb
            if bid_price >= ba:
                bid_price = ba - 1
            qty = min(limit - pos, max(8, gap // 4))
            pos = self._add_buy(orders, PEPPER, bid_price, qty, pos, limit)

        return orders

    # ------------------------------------------------------------------
    # OSMIUM
    # ------------------------------------------------------------------

    def trade_osmium(self, state: TradingState, od: OrderDepth, data):
        product = OSMIUM
        hard_limit = self.OSMIUM_HARD_LIMIT
        soft_limit = self.OSMIUM_SOFT_LIMIT
        pos = state.position.get(product, 0)

        bb, ba = self._best(od)
        if bb is None or ba is None:
            return []

        wb, wa, wb_sz, wa_sz = self._wall_levels(od)
        if wb is None or wa is None:
            return []

        mid = (bb + ba) / 2.0
        spread = ba - bb
        imb = self._imbalance(od)

        # ---- live wall_mid tracking ------------------------------------
        if wb_sz >= 5 and wa_sz >= 5:
            wall_mid = (wb + wa) / 2.0
        else:
            wall_mid = mid

        wmh = data["osmium_wallmid_hist"]
        wmh.append(wall_mid)
        if len(wmh) > self.OSMIUM_FV_HIST_LEN:
            del wmh[0]
        wallmid_smoothed = sum(wmh[-30:]) / min(30, len(wmh))

        # Blend live wall_mid anchor with constant 10000 (regularization)
        anchor = (self.OSMIUM_ANCHOR_BLEND * wallmid_smoothed
                  + (1 - self.OSMIUM_ANCHOR_BLEND) * 10000.0)

        # ---- v6's signal mix (kept because empirically wins) -----------
        short_ema = self._update_ema(data.get("osmium_short_ema"), mid, 0.10)
        long_ema = self._update_ema(data.get("osmium_long_ema"), mid, 0.03)
        data["osmium_short_ema"] = short_ema
        data["osmium_long_ema"] = long_ema

        mid_hist = data["osmium_mid_hist"]
        mid_hist.append(mid)
        if len(mid_hist) > 60:
            del mid_hist[0]
        recent = mid_hist[-30:] if len(mid_hist) >= 10 else mid_hist
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

        # ---- quote sizing & half-width ---------------------------------
        if spread >= 24:
            halfwidth = 4
            base_size = self.OSMIUM_BASE_SIZE + 2
        elif spread >= 18:
            halfwidth = 3
            base_size = self.OSMIUM_BASE_SIZE + 1
        else:
            halfwidth = 2
            base_size = self.OSMIUM_BASE_SIZE

        if abs(z) > 2.0:
            base_size = max(4, base_size - 3)

        orders: List[Order] = []
        signal = fv - mid

        # ---- aggressive take on strong signal --------------------------
        if signal > self.OSMIUM_TAKE_THRESHOLD and pos < soft_limit:
            for px, neg_qty in sorted(od.sell_orders.items()):
                qty = -neg_qty
                if px <= fv - self.OSMIUM_TAKE_EDGE and mid <= fv:
                    take = min(qty, 25, soft_limit - pos)
                    if take > 0:
                        pos = self._add_buy(orders, product, px, take, pos, hard_limit)
                else:
                    break

        if signal < -self.OSMIUM_TAKE_THRESHOLD and pos > -soft_limit:
            for px, qty in sorted(od.buy_orders.items(), reverse=True):
                if px >= fv + self.OSMIUM_TAKE_EDGE and mid >= fv:
                    take = min(qty, 25, soft_limit + pos)
                    if take > 0:
                        pos = self._add_sell(orders, product, px, take, pos, hard_limit)
                else:
                    break

        # ---- maker quotes ---------------------------------------------
        improve_bid = 1 if signal > 0.5 and spread >= 2 else 0
        improve_ask = 1 if signal < -0.5 and spread >= 2 else 0

        bid_price = min(bb + improve_bid, int(round(fv - halfwidth)))
        ask_price = max(ba - improve_ask, int(round(fv + halfwidth)))

        # Force-place inside the wall if conservative bid/ask is OUTSIDE
        # the wall. This puts us at the empty +1/-1 slot (data showed
        # zero trades there because nobody is currently quoting there).
        if wb_sz >= 10 and bid_price <= wb:
            bid_price = wb + 1
        if wa_sz >= 10 and ask_price >= wa:
            ask_price = wa - 1

        if bid_price >= ba:
            bid_price = ba - 1
        if ask_price <= bb:
            ask_price = bb + 1
        if ask_price - bid_price < 2:
            mid_q = (bid_price + ask_price) // 2
            bid_price = mid_q - 1
            ask_price = mid_q + 1

        buy_size = base_size
        sell_size = base_size

        if signal > 0.5:
            buy_size += 6
            sell_size = max(2, sell_size - 3)
        elif signal < -0.5:
            sell_size += 6
            buy_size = max(2, buy_size - 3)

        if pos > 50:
            buy_size = max(0, buy_size - 8)
            sell_size += 6
        elif pos > 25:
            buy_size = max(2, buy_size - 4)
            sell_size += 3
        elif pos < -50:
            sell_size = max(0, sell_size - 8)
            buy_size += 6
        elif pos < -25:
            sell_size = max(2, sell_size - 4)
            buy_size += 3

        buy_size = min(buy_size, max(0, hard_limit - pos))
        sell_size = min(sell_size, max(0, hard_limit + pos))

        pos = self._add_buy(orders, product, bid_price, buy_size, pos, hard_limit)
        pos = self._add_sell(orders, product, ask_price, sell_size, pos, hard_limit)

        return orders