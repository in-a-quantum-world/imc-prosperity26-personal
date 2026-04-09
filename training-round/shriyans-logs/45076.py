from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:
    def run(self, state: TradingState):

        # --- EMERALDS constants ---
        EM_POSITION_LIMIT       = 80
        EM_SOFT_LIMIT           = 60
        EM_FAIR_VALUE           = 10000
        EM_IMBALANCE_THRESHOLD  = 0.2
        EM_INVENTORY_SKEW       = 2

        # --- TOMATOES constants ---
        TOM_POSITION_LIMIT      = 80
        TOM_FAST_WINDOW          = 10   # short-term EMA window
        TOM_SLOW_WINDOW          = 25   # long-term EMA window
        TOM_ROC_PERIOD           = 10   # rate-of-change lookback (shortened)
        TOM_MOMENTUM_THRESHOLD   = 0.001  # minimum ROC to signal (0.1%)
        TOM_STRONG_TREND_THRESH  = 0.002  # strong trend threshold (0.2%)
        TOM_MAX_POSITION         = 60    # max position during neutral

        # --- deserialise state ---
        history = json.loads(state.traderData) if state.traderData else {}
        result  = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order]     = []

            best_bid = max(order_depth.buy_orders.keys())  if order_depth.buy_orders  else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

            if best_bid is None or best_ask is None:
                continue

            current_position = state.position.get(product, 0)

            if product == "EMERALDS":
                buy_capacity  = EM_SOFT_LIMIT - current_position
                sell_capacity = EM_SOFT_LIMIT + current_position

                # ---------- ORDER BOOK IMBALANCE SIGNAL ----------

                total_bid_vol   = sum(order_depth.buy_orders.values())
                total_ask_vol   = abs(sum(order_depth.sell_orders.values()))

                imbalance_ratio = (
                    (total_bid_vol - total_ask_vol)
                    / (total_bid_vol + total_ask_vol)
                )

                # ---------- INVENTORY SIGNAL ----------

                inventory_bias  = current_position / EM_POSITION_LIMIT

                # ---------- BASE MARKET MAKING PRICES ----------

                bid_price = best_bid + 1
                ask_price = best_ask - 1

                # ---------- APPLY ORDER BOOK PRESSURE SKEW ----------

                if imbalance_ratio > EM_IMBALANCE_THRESHOLD:
                    bid_price += 1
                    ask_price += 1
                elif imbalance_ratio < -EM_IMBALANCE_THRESHOLD:
                    bid_price -= 1
                    ask_price -= 1

                # ---------- APPLY INVENTORY SKEW ----------

                inventory_adjustment = int(inventory_bias * EM_INVENTORY_SKEW)
                bid_price -= inventory_adjustment
                ask_price -= inventory_adjustment

                # ---------- CLAMP AGAINST FAIR VALUE SAFETY ----------

                bid_price = min(bid_price, EM_FAIR_VALUE)
                ask_price = max(ask_price, EM_FAIR_VALUE)

                # ---------- PLACE ORDERS ----------

                if buy_capacity > 0:
                    orders.append(Order(product, bid_price,  buy_capacity))
                if sell_capacity > 0:
                    orders.append(Order(product, ask_price, -sell_capacity))

            # Momentum-based tomato strategy
            # Uses EMA crossover + Rate-of-Change for robust momentum signals
            elif product == "TOMATOES":
                mid_price = (best_bid + best_ask) / 2

                prices = history.get(product, [])
                prices.append(mid_price)
                if len(prices) > TOM_SLOW_WINDOW + TOM_ROC_PERIOD:
                    prices = prices[-(TOM_SLOW_WINDOW + TOM_ROC_PERIOD):]
                history[product] = prices

                if len(prices) < TOM_SLOW_WINDOW + TOM_ROC_PERIOD:
                    result[product] = orders
                    continue

                # --- Compute EMA ---
                def ema(data, window):
                    k = 2 / (window + 1)
                    ema_val = sum(data[:window]) / window
                    for price in data[window:]:
                        ema_val = price * k + ema_val * (1 - k)
                    return ema_val

                fast_ema = ema(prices, TOM_FAST_WINDOW)
                slow_ema = ema(prices, TOM_SLOW_WINDOW)

                # --- Compute Rate-of-Change (momentum) ---
                roc = (prices[-1] - prices[-TOM_ROC_PERIOD]) / prices[-TOM_ROC_PERIOD]

                # --- Compute EMA crossover signal ---
                ema_diff = (fast_ema - slow_ema) / slow_ema

                # --- Directional signal ---
                buy_capacity  = TOM_POSITION_LIMIT - current_position
                sell_capacity = TOM_POSITION_LIMIT + current_position

                # Strong uptrend: positive ROC + EMA crossover bullish
                if roc > TOM_STRONG_TREND_THRESH and ema_diff > 0 and buy_capacity > 0:
                    ask_volume = abs(order_depth.sell_orders[best_ask])
                    qty = min(ask_volume, buy_capacity)
                    orders.append(Order(product, best_ask - 1, qty))

                # Strong downtrend: negative ROC + EMA crossover bearish
                elif roc < -TOM_STRONG_TREND_THRESH and ema_diff < 0 and sell_capacity > 0:
                    bid_volume = order_depth.buy_orders[best_bid]
                    qty = min(bid_volume, sell_capacity)
                    orders.append(Order(product, best_bid + 1, -qty))

                # Mild momentum (weaker signal, cap position)
                elif roc > TOM_MOMENTUM_THRESHOLD and buy_capacity > 0 and current_position < TOM_MAX_POSITION:
                    ask_volume = abs(order_depth.sell_orders[best_ask])
                    qty = min(ask_volume, min(buy_capacity, TOM_MAX_POSITION - current_position))
                    orders.append(Order(product, best_ask - 1, qty))

                elif roc < -TOM_MOMENTUM_THRESHOLD and sell_capacity > 0 and current_position > -TOM_MAX_POSITION:
                    bid_volume = order_depth.buy_orders[best_bid]
                    qty = min(bid_volume, min(sell_capacity, TOM_MAX_POSITION + current_position))
                    orders.append(Order(product, best_bid + 1, -qty))

                # --- Fallback: market-make if no momentum signal ---
                else:
                    mm_bid = best_bid + 1
                    mm_ask = best_ask - 1
                    # Slight skew based on existing position
                    if current_position > 0:
                        mm_bid -= 1  # less aggressive on bid if long
                    elif current_position < 0:
                        mm_ask += 1  # less aggressive on ask if short
                    if buy_capacity > 0:
                        orders.append(Order(product, mm_bid, buy_capacity))
                    if sell_capacity > 0:
                        orders.append(Order(product, mm_ask, -sell_capacity))

            result[product] = orders

        traderData = json.dumps(history)
        return result, 0, traderData