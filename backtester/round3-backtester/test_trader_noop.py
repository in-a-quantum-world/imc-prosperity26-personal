"""No-op trader for sanity checking the backtester. Expected PnL: 0 everywhere."""
from datamodel import TradingState


class Trader:
    def run(self, state: TradingState):
        return {}, 0, ""
