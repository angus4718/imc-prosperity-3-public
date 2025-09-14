import math
from abc import abstractmethod, ABC
from collections import deque
from copy import deepcopy
from math import exp, log, sqrt
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, TypeAlias
import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class MarketUtils:
    """Holds utility functions used across strategies."""

    def safe_best_bid(self, order_depth) -> Optional[int]:
        if not order_depth or not order_depth.buy_orders:
            return None
        return max(order_depth.buy_orders.keys())

    def safe_best_ask(self, order_depth) -> Optional[int]:
        if not order_depth or not order_depth.sell_orders:
            return None
        return min(order_depth.sell_orders.keys())

    def has_liquidity(self, order_depth) -> bool:
        return bool(order_depth and order_depth.buy_orders and order_depth.sell_orders)

    def rolling_mean_std(self, values: List[float], period: int) -> Tuple[float, float]:
        window = values[-period:]
        n = len(window)
        if n == 0:
            return 0.0, 0.0
        m = sum(window) / n
        var = sum((x - m) ** 2 for x in window) / n
        return m, math.sqrt(var)


class InteractionBlocks:
    """Wraps strategy helper functions: mean reversion, taking, making, zero-EV."""

    def __init__(self, utils: MarketUtils):
        self.u = utils

    def mean_reversion_taker(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        price_array: List[float],
        fair_price: float,
        period: int,
        z_score_threshold: float,
        fixed_threshold: float,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        if len(price_array) < period:
            return

        rolling_mean = sum(price_array[-period:]) / period
        rolling_std = (sum((x - rolling_mean) ** 2 for x in price_array[-period:]) / period) ** 0.5

        if rolling_std == 0:
            return

        deviation = fair_price - rolling_mean
        z_score = deviation / rolling_std

        if z_score < -z_score_threshold and deviation < -fixed_threshold and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            amount_to_buy = min(
                -order_depth.sell_orders[best_ask],
                limit - position - parent.total_buying_amount,
            )
            if amount_to_buy > 0 and best_ask > 0:
                parent.buy(best_ask, amount_to_buy)
                parent.total_buying_amount += amount_to_buy

        elif z_score > z_score_threshold and deviation > fixed_threshold and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            amount_to_sell = min(
                order_depth.buy_orders[best_bid],
                limit + position - parent.total_selling_amount,
            )
            if amount_to_sell > 0 and best_bid > 0:
                parent.sell(best_bid, amount_to_sell)
                parent.total_selling_amount += amount_to_sell

    def market_taking_strategy(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        fair_buying_price: float,
        fair_selling_price: float,
        max_size: int,
    ) -> Tuple[int, int]:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal
        sizes = [0, 0]

        # Market taking: selling orders
        max_buy_amount = max_size
        if order_depth.sell_orders:
            asks = sorted(order_depth.sell_orders.keys())
            for best_buying_price in asks:
                if best_buying_price <= fair_buying_price:
                    best_buying_amount = -order_depth.sell_orders[best_buying_price]
                    amount_to_buy = min(
                        best_buying_amount, limit - position - parent.total_buying_amount, max_buy_amount
                    )
                    if amount_to_buy > 0:
                        parent.buy(best_buying_price, amount_to_buy)
                        sizes[0] += amount_to_buy
                        max_buy_amount -= amount_to_buy

        # Market taking: buying orders
        max_sell_amount = max_size
        if order_depth.buy_orders:
            bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            for best_selling_price in bids:
                if best_selling_price >= fair_selling_price:
                    best_selling_amount = order_depth.buy_orders[best_selling_price]
                    amount_to_sell = min(
                        best_selling_amount, limit + position - parent.total_selling_amount, max_sell_amount
                    )
                    if amount_to_sell > 0:
                        parent.sell(best_selling_price, amount_to_sell)
                        sizes[1] += amount_to_sell
                        max_sell_amount -= amount_to_sell

        return sizes[0], sizes[1]

    def market_making_strategy(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        zero_ev_bid: float,
        zero_ev_ask: float,
        pos_ev_bid: float,
        pos_ev_ask: float,
        max_bid_size: int,
        max_ask_size: int,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        # Market making: buy orders
        remaining_buy_capacity = min(limit - position - parent.total_buying_amount, max_bid_size)
        if remaining_buy_capacity > 0:
            max_buy_price = (
                max(
                    [price for price in order_depth.buy_orders.keys() if price < zero_ev_bid],
                    default=pos_ev_bid,
                )
                + 1
            )
            buy_price = min(max_buy_price, pos_ev_bid)
            parent.buy(buy_price, remaining_buy_capacity)

        # Market making: sell orders
        remaining_sell_capacity = min(limit + position - parent.total_selling_amount, max_ask_size)
        if remaining_sell_capacity > 0:
            min_sell_price = (
                min(
                    [price for price in order_depth.sell_orders.keys() if price > zero_ev_ask],
                    default=pos_ev_ask,
                )
                - 1
            )
            sell_price = max(min_sell_price, pos_ev_ask)
            parent.sell(sell_price, remaining_sell_capacity)

    def zero_ev_trades(
        self,
        state,  # TradingState
        parent,  # Strategy
        symbol: str,
        limit: int,
        fair_buying_price: float,
        fair_selling_price: float,
    ) -> None:
        position = state.position.get(symbol, 0)
        order_depth = parent.order_depth_internal

        # Zero EV sell trades
        if position > 0 and fair_selling_price in order_depth.buy_orders:
            amount_to_sell = min(
                position,
                order_depth.buy_orders[fair_selling_price],
                limit + position - parent.total_selling_amount,
            )
            if amount_to_sell > 0:
                parent.sell(fair_selling_price, amount_to_sell)

        # Zero EV buy trades
        elif position < 0 and fair_buying_price in order_depth.sell_orders:
            amount_to_buy = min(
                -position,
                -order_depth.sell_orders[fair_buying_price],
                limit - position - parent.total_buying_amount,
            )
            if amount_to_buy > 0:
                parent.buy(fair_buying_price, amount_to_buy)


class SignalSnoopers:

    def get_olivia_signal(self, product, state) -> int:
        buy_bots = [t.buyer for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])]
        sell_bots = [t.seller for t in state.own_trades.get(product, []) + state.market_trades.get(product, [])]
        if "Olivia" in buy_bots:
            return 1
        if "Olivia" in sell_bots:
            return -1
        return 0


class Strategy(ABC):
    """Base class unchanged in logic; provides order handling and internal book simulation."""

    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit
        self.total_buying_amount = 0
        self.total_selling_amount = 0
        self.order_depth_internal = None
        self.orders = []
        self.conversions = 0

    @abstractmethod
    def act(self, state) -> None:
        raise NotImplementedError()

    def run(self, state) -> Tuple[List, int]:
        self.total_buying_amount = 0
        self.total_selling_amount = 0
        self.orders = []
        self.conversions = 0

        self.order_depth_internal = deepcopy(state.order_depths.get(self.symbol, []))

        self.act(state)

        return self.orders, self.conversions

    def popular_price_calculator(self, order_depth) -> float:
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders and not sell_orders:
            return 0
        elif not buy_orders:
            popular_buying_price = min(sell_orders.keys()) - 1
        else:
            popular_buying_price = max(buy_orders.keys())

        if not sell_orders:
            popular_selling_price = max(buy_orders.keys()) + 1
        else:
            popular_selling_price = min(sell_orders.keys())

        popular_price = (popular_buying_price + popular_selling_price) / 2

        return popular_price

    def buy(self, price: int, quantity: int) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order(self.symbol, price, quantity))
        self.total_buying_amount += quantity

        remaining_quantity = quantity
        while (
            price >= min(self.order_depth_internal.sell_orders.keys(), default=float("inf")) and remaining_quantity > 0
        ):
            ask_price = min(self.order_depth_internal.sell_orders.keys())
            ask_size = -self.order_depth_internal.sell_orders[ask_price]
            if ask_size > remaining_quantity:
                self.order_depth_internal.sell_orders[ask_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                self.order_depth_internal.sell_orders.pop(ask_price)
                remaining_quantity -= ask_size

    def sell(self, price: int, quantity: int) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order(self.symbol, price, -quantity))
        self.total_selling_amount += quantity

        remaining_quantity = quantity
        while (
            price <= max(self.order_depth_internal.buy_orders.keys(), default=float("-inf")) and remaining_quantity > 0
        ):
            bid_price = max(self.order_depth_internal.buy_orders.keys())
            bid_size = self.order_depth_internal.buy_orders[bid_price]
            if bid_size > remaining_quantity:
                self.order_depth_internal.buy_orders[bid_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                self.order_depth_internal.buy_orders.pop(bid_price)
                remaining_quantity -= bid_size

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class BlackScholes:

    def black_scholes_call(self, spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    def delta(self, spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        return NormalDist().cdf(d1)

    def implied_volatility(self, call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class KelpStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, deflection_threshold: float, utils: MarketUtils, blocks: InteractionBlocks):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.deflection_threshold = deflection_threshold
        self.price_array = []
        self.price_array_max_length = 10
        self.fair_price = 0.0
        self.u = utils
        self.b = blocks

    def act(self, state) -> None:
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return

        self.fair_price = self.popular_price_calculator(order_depth)
        zero_ev_bid, zero_ev_ask = math.floor(self.fair_price), math.ceil(self.fair_price)
        pos_ev_bid = zero_ev_bid if zero_ev_bid < self.fair_price else zero_ev_bid - 1
        pos_ev_ask = zero_ev_ask if zero_ev_ask > self.fair_price else zero_ev_ask + 1

        # Deflection mechanism
        if len(self.price_array) > 0:
            change_in_fair = self.fair_price - self.price_array[-1]
            if change_in_fair > self.deflection_threshold:
                # Deduct 100 from pos_ev_bid so that the algo only sells
                pos_ev_bid -= 100
            elif change_in_fair < -self.deflection_threshold:
                # Add 100 to pos_ev_ask so that the algo only buys
                pos_ev_ask += 100

        self.b.zero_ev_trades(state, self, self.symbol, self.limit, zero_ev_bid, zero_ev_ask)
        self.b.market_taking_strategy(state, self, self.symbol, self.limit, pos_ev_bid, pos_ev_ask, self.limit)
        self.b.market_making_strategy(
            state,
            self,
            self.symbol,
            self.limit,
            zero_ev_bid,
            zero_ev_ask,
            pos_ev_bid,
            pos_ev_ask,
            self.limit,
            self.limit,
        )

    def save(self) -> JSON:
        self.price_array.append(self.fair_price)
        if len(self.price_array) > self.price_array_max_length:
            self.price_array.pop(0)
        return jsonpickle.encode(self.price_array)

    def load(self, data) -> None:
        self.price_array = jsonpickle.decode(data)


class SquidInkStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, utils: MarketUtils, blocks: InteractionBlocks, snoop: SignalSnoopers):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit

        self.price_array = []
        self.price_array_max_length = 100
        self.fair_price = 0.0
        self.u = utils
        self.b = blocks
        self.s = snoop

    def act(self, state) -> None:
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return

        self.fair_price = self.popular_price_calculator(order_depth)
        olivia_signal = self.s.get_olivia_signal(self.symbol, state)
        if olivia_signal == 1:
            self.b.market_taking_strategy(state, self, self.symbol, self.limit, 99999, 99999, self.limit)
            return
        elif olivia_signal == -1:
            self.b.market_taking_strategy(state, self, self.symbol, self.limit, 1, 1, self.limit)
            return

        self.b.mean_reversion_taker(state, self, self.symbol, self.limit, self.price_array, self.fair_price, 100, 0, 30)

    def save(self) -> JSON:
        self.price_array.append(self.fair_price)
        if len(self.price_array) > self.price_array_max_length:
            self.price_array.pop(0)
        return jsonpickle.encode(self.price_array)

    def load(self, data) -> None:
        self.price_array = jsonpickle.decode(data)


class CroissantStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, snoop: SignalSnoopers, blocks: InteractionBlocks):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.s = snoop
        self.b = blocks

    def act(self, state) -> None:
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return

        olivia_signal = self.s.get_olivia_signal(self.symbol, state)
        if olivia_signal == 1:
            self.b.market_taking_strategy(state, self, self.symbol, self.limit, 99999, 99999, self.limit)
            return
        elif olivia_signal == -1:
            self.b.market_taking_strategy(state, self, self.symbol, self.limit, 1, 1, self.limit)
            return


class RainforestResinStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, blocks: InteractionBlocks):
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.market_taking = None
        self.zero_ev = None
        self.market_making = None
        self.b = blocks

    def act(self, state) -> None:
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return

        zero_ev_bid, zero_ev_ask = 10000, 10000
        pos_ev_bid, pos_ev_ask = 9999, 10001
        self.b.market_taking_strategy(state, self, self.symbol, self.limit, pos_ev_bid, pos_ev_ask, self.limit)
        self.b.zero_ev_trades(state, self, self.symbol, self.limit, zero_ev_bid, zero_ev_ask)
        self.b.market_making_strategy(
            state,
            self,
            self.symbol,
            self.limit,
            zero_ev_bid,
            zero_ev_ask,
            pos_ev_bid,
            pos_ev_ask,
            self.limit,
            self.limit,
        )


class PicnicBasketStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        limit: int,
        position_threshold: int,
        aggressive_adj: int,
        aggressive_deflection_adj: int,
        normal_deflection_adj: int,
        default_size: int,
        utils: MarketUtils,
        blocks: InteractionBlocks,
    ) -> None:
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.position_threshold = position_threshold
        self.aggressive_adj = aggressive_adj
        self.aggressive_deflection_adj = aggressive_deflection_adj
        self.normal_deflection_adj = normal_deflection_adj
        self.default_size = default_size
        self.u = utils
        self.b = blocks

    def act(self, state) -> None:
        if any(
            symbol not in state.order_depths
            for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        ):
            return
        if any(
            len(state.order_depths[symbol].buy_orders) == 0 or len(state.order_depths[symbol].sell_orders) == 0
            for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", "PICNIC_BASKET2"]
        ):
            return

        position = state.position.get(self.symbol, 0)

        croissants_mid = self.get_mid(state.order_depths["CROISSANTS"], "impact_weighted_mid_price2")
        jams_mid = self.get_mid(state.order_depths["JAMS"], "weighted_mid_price")
        djembes_mid = self.get_mid(state.order_depths["DJEMBES"], "weighted_mid_price")
        picnic_basket1_mid = self.get_mid(state.order_depths["PICNIC_BASKET1"], "popular_simple_mid_price")
        picnic_basket2_mid = self.get_mid(state.order_depths["PICNIC_BASKET2"], "weighted_mid_price")

        spread_1 = picnic_basket1_mid - (croissants_mid * 6 + jams_mid * 3 + djembes_mid * 1)
        spread_2 = picnic_basket2_mid - (croissants_mid * 4 + jams_mid * 2)
        spread_3 = picnic_basket1_mid - (picnic_basket2_mid * 3 / 2 + djembes_mid)
        desired_position = self.get_desired_position(spread_1, spread_2, spread_3)
        amount_to_buy = max(0, desired_position - position)
        amount_to_sell = max(0, position - desired_position)

        self.fair_price = {"PICNIC_BASKET1": picnic_basket1_mid, "PICNIC_BASKET2": picnic_basket2_mid}[self.symbol]
        zero_ev_bid, zero_ev_ask = math.floor(self.fair_price), math.ceil(self.fair_price)

        if amount_to_buy > 0:
            if amount_to_buy > self.position_threshold:
                bid = zero_ev_bid + self.aggressive_adj
                ask = zero_ev_ask + self.aggressive_deflection_adj
                sent_sizes = self.b.market_taking_strategy(state, self, self.symbol, self.limit, bid, ask, amount_to_buy)
                self.b.market_making_strategy(
                    state,
                    self,
                    self.symbol,
                    self.limit,
                    bid,
                    ask,
                    bid,
                    ask,
                    max(0, amount_to_buy - sent_sizes[0]),
                    self.default_size,
                )
            else:
                bid = zero_ev_bid
                ask = zero_ev_ask + self.normal_deflection_adj
                sent_sizes = self.b.market_taking_strategy(state, self, self.symbol, self.limit, bid, ask, amount_to_buy)
                self.b.market_making_strategy(
                    state,
                    self,
                    self.symbol,
                    self.limit,
                    bid,
                    ask,
                    bid,
                    ask,
                    max(0, amount_to_buy - sent_sizes[0]),
                    self.default_size,
                )

        elif amount_to_sell > 0:
            if amount_to_sell > self.position_threshold:
                bid = zero_ev_bid - self.aggressive_deflection_adj
                ask = zero_ev_ask - self.aggressive_adj
                sent_sizes = self.b.market_taking_strategy(state, self, self.symbol, self.limit, bid, ask, amount_to_sell)
                self.b.market_making_strategy(
                    state,
                    self,
                    self.symbol,
                    self.limit,
                    bid,
                    ask,
                    bid,
                    ask,
                    self.default_size,
                    max(0, amount_to_sell - sent_sizes[1]),
                )
            else:
                bid = zero_ev_bid - self.normal_deflection_adj
                ask = zero_ev_ask
                sent_sizes = self.b.market_taking_strategy(state, self, self.symbol, self.limit, bid, ask, amount_to_sell)
                self.b.market_making_strategy(
                    state,
                    self,
                    self.symbol,
                    self.limit,
                    bid,
                    ask,
                    bid,
                    ask,
                    self.default_size,
                    max(0, amount_to_sell - sent_sizes[1]),
                )

        else:
            bid, ask = zero_ev_bid, zero_ev_ask
            sent_sizes = self.b.market_taking_strategy(state, self, self.symbol, self.limit, bid, ask, self.default_size)
            self.b.market_making_strategy(
                state,
                self,
                self.symbol,
                self.limit,
                bid,
                ask,
                bid,
                ask,
                self.default_size,
                self.default_size,
            )

        logger.print(self.symbol, spread_1, spread_2, desired_position, position, amount_to_buy, amount_to_sell)

    def get_mid(self, order_depth, method: str) -> float:
        bid_prices, ask_prices, bid_volumes, ask_volumes = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        for i, (price, volume) in enumerate(sorted(order_depth.buy_orders.items(), reverse=True)[:3]):
            bid_prices[i] = price
            bid_volumes[i] = volume
        for i, (price, volume) in enumerate(sorted(order_depth.sell_orders.items(), reverse=False)[:3]):
            ask_prices[i] = price
            ask_volumes[i] = -volume

        if method == "impact_weighted_mid_price2":
            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)
            weighted_bid = sum(price * volume for price, volume in zip(bid_prices, bid_volumes)) / total_bid_volume
            weighted_ask = sum(price * volume for price, volume in zip(ask_prices, ask_volumes)) / total_ask_volume
            return (weighted_bid * total_ask_volume + weighted_ask * total_bid_volume) / (
                total_bid_volume + total_ask_volume
            )

        elif method == "weighted_mid_price":
            return (bid_prices[0] * ask_volumes[0] + ask_prices[0] * bid_volumes[0]) / (bid_volumes[0] + ask_volumes[0])

        elif method == "popular_simple_mid_price":
            popular_bid = bid_prices[bid_volumes.index(max(bid_volumes))]
            popular_ask = ask_prices[ask_volumes.index(max(ask_volumes))]
            return (popular_bid + popular_ask) / 2

        else:
            raise Exception()

    def get_desired_position(self, spread1, spread2, spread3):
        if self.symbol == "PICNIC_BASKET1":
            res = -(math.tanh(spread1 / 85) + math.tanh(spread3 / 95))
            res = max(min(res, 1), -1) * self.limit
            return round(res)

        elif self.symbol == "PICNIC_BASKET2":
            res = -(math.tanh(spread2 / 65))
            res = max(min(res, 1), -1) * self.limit
            return round(res)

        else:
            return 0


class OptionStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        limit: int,
        window: int,
        z_score_threshold: float,
        strike: int,
        utils: MarketUtils,
        blocks: InteractionBlocks,
    ) -> None:
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.window = window
        self.threshold = 0.01
        self.strike = strike
        self.z_score_threshold = z_score_threshold
        self.price_array = []
        self.underlying_current_orders = 0

        self.options_model = BlackScholes()

        self.implied_vols = []
        self.deltas = []

        self.underlying_price_history = []
        self.timestamps_per_year = 365e6
        self.days_left = 3

        # Volatility parameters (coefficients for IV curve fitting)
        self.ask_params = {"a": 0.2878490683651303, "b": -0.0009201058370376721, "c": 0.1499510693474374}
        self.bid_params = {"a": 0.1850111314490967, "b": 0.0008529599232086579, "c": 0.14879176125529844}

        self.u = utils
        self.b = blocks

    def act(self, state) -> None:
        self.underlying_current_orders = 0
        order_depth = state.order_depths.get(self.symbol, None)
        if not order_depth:
            return

        # Update underlying price and calculate implied volatility
        underlying_symbol = "VOLCANIC_ROCK"
        underlying_order_depth = state.order_depths.get(underlying_symbol, None)
        if not underlying_order_depth or not underlying_order_depth.sell_orders or not underlying_order_depth.buy_orders:
            return
        underlying_bid = min(underlying_order_depth.buy_orders.keys())
        underlying_ask = max(underlying_order_depth.sell_orders.keys())
        underlying_price = (underlying_ask + underlying_bid) / 2
        self.underlying_price_history.append(underlying_price)
        self.underlying_price_history = self.underlying_price_history[-5:]
        underlying_price = self.underlying_price_history[-1]
        tte = (self.days_left / 365) - state.timestamp / self.timestamps_per_year
        moneyness = np.log(self.strike / underlying_price) / np.sqrt(tte)

        theoretical_ask_price = None
        theoretical_bid_price = None

        # Checking ask level
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            theoretical_iv_ask = self.ask_params["c"] + self.ask_params["b"] * moneyness + self.ask_params["a"] * moneyness**2
            theoretical_ask_price = self.options_model.black_scholes_call(underlying_price, self.strike, tte, theoretical_iv_ask)

        # Checking bid level
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            theoretical_iv_bid = self.bid_params["c"] + self.bid_params["b"] * moneyness + self.bid_params["a"] * moneyness**2
            theoretical_bid_price = self.options_model.black_scholes_call(underlying_price, self.strike, tte, theoretical_iv_bid)

        # Update price array
        if not order_depth.sell_orders:
            mid_price = best_bid + 1
            theoretical_mid_price = theoretical_bid_price + 1
            theoretical_iv_mid = theoretical_iv_bid
        elif not order_depth.buy_orders:
            mid_price = best_ask - 1
            theoretical_mid_price = theoretical_ask_price - 1
            theoretical_iv_mid = theoretical_iv_ask
        else:
            mid_price = (best_ask + best_bid) / 2
            theoretical_mid_price = (theoretical_ask_price + theoretical_bid_price) / 2
            theoretical_iv_mid = (theoretical_iv_ask + theoretical_iv_bid) / 2
        self.price_array.append(mid_price)
        if len(self.price_array) < self.window:
            return

        self.price_array = self.price_array[-self.window:]
        self.b.mean_reversion_taker(
            state, self, self.symbol, self.limit, self.price_array, theoretical_mid_price, self.window, self.z_score_threshold, 1
        )

        current_position = state.position.get(self.symbol, 0) + self.total_buying_amount - self.total_selling_amount
        delta = self.options_model.delta(underlying_price, self.strike, tte, theoretical_iv_mid)
        total_delta = int(current_position * delta)
        position_to_fill = total_delta - state.position.get(underlying_symbol, 0) - self.underlying_current_orders

        if position_to_fill > 0:
            size = min(position_to_fill, self.limit)
            self.buy_underlying(underlying_ask, size, underlying_order_depth)
        elif position_to_fill < 0:
            size = min(-position_to_fill, self.limit)
            self.sell_underlying(underlying_bid, size, underlying_order_depth)

    def buy_underlying(self, price: int, quantity: int, underlying_order_depth) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order("VOLCANIC_ROCK", price, quantity))
        self.underlying_current_orders += quantity

        remaining_quantity = quantity

        while (
            price >= min(underlying_order_depth.sell_orders.keys(), default=float("inf")) and remaining_quantity > 0
        ):
            ask_price = min(underlying_order_depth.sell_orders.keys())
            ask_size = -underlying_order_depth.sell_orders[ask_price]
            if ask_size > remaining_quantity:
                underlying_order_depth.sell_orders[ask_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                underlying_order_depth.sell_orders.pop(ask_price)
                remaining_quantity -= ask_size

    def sell_underlying(self, price: int, quantity: int, underlying_order_depth) -> None:
        assert isinstance(price, int)
        assert isinstance(quantity, int)
        assert price > 0
        assert quantity > 0

        self.orders.append(Order("VOLCANIC_ROCK", price, -quantity))
        self.underlying_current_orders -= quantity

        remaining_quantity = quantity
        while (
            price <= max(underlying_order_depth.buy_orders.keys(), default=float("-inf")) and remaining_quantity > 0
        ):
            bid_price = max(underlying_order_depth.buy_orders.keys())
            bid_size = underlying_order_depth.buy_orders[bid_price]
            if bid_size > remaining_quantity:
                underlying_order_depth.buy_orders[bid_price] -= remaining_quantity
                remaining_quantity = 0
            else:
                underlying_order_depth.buy_orders.pop(bid_price)
                remaining_quantity -= bid_size


class MacaronsStrategy(Strategy):
    def __init__(self, symbol: str, limit: int, conversion_limit: int, sunlight_threshold: float, z_score_threshold: float) -> None:
        super().__init__(symbol, limit)
        self.symbol = symbol
        self.limit = limit
        self.sunlight_threshold = sunlight_threshold
        self.conversion_limit = conversion_limit
        self.previous_sunlight = None
        self.previous_sugar = None
        self.price_array = []
        self.fair_price = 0
        self.current_trend_sunlight = None
        self.current_trend_sugar = None

    def act(self, state) -> None:
        position = state.position.get(self.symbol, 0)

        # Get observations
        obs = state.observations.conversionObservations.get(self.symbol, None)
        if obs is None:
            return
        order_depth = state.order_depths.get(self.symbol, None)
        if order_depth is None:
            return

        # Extract sugar price and sunlight index
        sugar_price = obs.sugarPrice
        sunlight = obs.sunlightIndex

        # Initialize previous values if not set
        if self.previous_sugar is None:
            self.previous_sugar = sugar_price
        if self.previous_sunlight is None:
            self.previous_sunlight = sunlight

        # Update rolling price array and calculate Z-score
        self.price_array.append(sugar_price)
        if len(self.price_array) > 100:  # Keep a rolling window of 100 prices
            self.price_array.pop(0)

        # Calculate fair price
        self.fair_price = self.popular_price_calculator(order_depth)

        # Determine buy/sell amounts based on position limits
        buy_amount = min(self.limit - position, self.conversion_limit)
        sell_amount = min(self.limit + position, self.conversion_limit)

        # Local and foreign price calculations
        foreign_ask = obs.askPrice + obs.transportFees - obs.importTariff
        foreign_bid = obs.bidPrice - obs.transportFees - obs.exportTariff
        local_ask = min(order_depth.sell_orders.keys())
        local_bid = max(order_depth.buy_orders.keys())
        local_ask_volume = abs(order_depth.sell_orders.get(local_ask, 0))
        local_bid_volume = abs(order_depth.buy_orders.get(local_bid, 0))

        # Detect sunlight trend
        sunlight_change = sunlight - self.previous_sunlight
        if sunlight_change > 0:
            self.current_trend_sunlight = 'uptrend'
        elif sunlight_change < 0:
            self.current_trend_sunlight = 'downtrend'
        # If sunlight_change == 0, maintain the current trend

        # Detect sugar price trend
        sugar_change = sugar_price - self.previous_sugar
        if sugar_change > 0:
            self.current_trend_sugar = 'uptrend'
        elif sugar_change < 0:
            self.current_trend_sugar = 'downtrend'
        # If sugar_change == 0, maintain the current trend

        if state.timestamp > 999000:  # Start unwinding positions aggressively near the end of the day
            # Gradually reduce position to zero
            if position > 0:  # If long position, sell to reduce it
                if local_bid > foreign_bid:
                    self.sell(local_bid, sell_amount)
                else:
                    convert_amount = min(abs(position), sell_amount, self.conversion_limit)
                    self.convert(-convert_amount)
                    remaining_amount = sell_amount - convert_amount
                    if remaining_amount > 0:
                        self.sell(local_bid, min(remaining_amount, local_bid_volume))
            elif position < 0:  # If short position, buy to reduce it
                if local_ask < foreign_ask:
                    self.buy(local_ask, buy_amount)
                else:
                    convert_amount = min(abs(position), buy_amount, self.conversion_limit)
                    self.convert(convert_amount)
                    remaining_amount = buy_amount - convert_amount
                    if remaining_amount > 0:
                        self.buy(local_ask, min(remaining_amount, local_ask_volume))
            return

        # Act based on trends when sunlight is below the threshold
        elif sunlight <= self.sunlight_threshold:
            if self.current_trend_sunlight == 'uptrend' and sell_amount > 0 and self.current_trend_sugar == 'downtrend':
                # Uptrend in sunlight and downtrend in sugar price: Sell
                if position > 0:
                    if local_bid > foreign_bid:
                        self.sell(local_bid, sell_amount)
                    else:
                        convert_amount = min(abs(position), sell_amount, self.conversion_limit)
                        self.convert(-convert_amount)
                        remaining_amount = sell_amount - convert_amount
                        if remaining_amount > 0:
                            self.sell(local_bid, min(remaining_amount, local_bid_volume))
                else:
                    self.sell(local_bid, sell_amount)

            elif self.current_trend_sunlight == 'downtrend' and buy_amount > 0 and self.current_trend_sugar == 'uptrend':
                # Downtrend in sunlight and uptrend in sugar price: Buy
                if position < 0:
                    if local_ask < foreign_ask:
                        self.buy(local_ask, buy_amount)
                    else:
                        convert_amount = min(abs(position), buy_amount, self.conversion_limit)
                        self.convert(convert_amount)
                        remaining_amount = buy_amount - convert_amount
                        if remaining_amount > 0:
                            self.buy(local_ask, min(remaining_amount, local_ask_volume))
                else:
                    self.buy(local_ask, buy_amount)

        # Update previous values
        self.previous_sunlight = sunlight
        self.previous_sugar = sugar_price


class Trader:

    def __init__(self):
        self.utils = MarketUtils()
        self.blocks = InteractionBlocks(self.utils)
        self.snoop = SignalSnoopers()

        self.limit = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75,
        }

        self.strategies = {
            "RAINFOREST_RESIN": RainforestResinStrategy(
                symbol="RAINFOREST_RESIN", limit=self.limit["RAINFOREST_RESIN"], blocks=self.blocks
            ),
            "KELP": KelpStrategy(symbol="KELP", limit=self.limit["KELP"], deflection_threshold=0.5, utils=self.utils, blocks=self.blocks),
            "SQUID_INK": SquidInkStrategy(
                symbol="SQUID_INK",
                limit=self.limit["SQUID_INK"],
                utils=self.utils,
                blocks=self.blocks,
                snoop=self.snoop,
            ),
            "CROISSANTS": CroissantStrategy(
                symbol="CROISSANTS",
                limit=self.limit["CROISSANTS"],
                snoop=self.snoop,
                blocks=self.blocks,
            ),
            "PICNIC_BASKET1": PicnicBasketStrategy(
                symbol="PICNIC_BASKET1",
                limit=self.limit["PICNIC_BASKET1"],
                position_threshold=60,
                aggressive_adj=3,
                aggressive_deflection_adj=100,
                normal_deflection_adj=100,
                default_size=3,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "PICNIC_BASKET2": PicnicBasketStrategy(
                symbol="PICNIC_BASKET2",
                limit=self.limit["PICNIC_BASKET2"],
                position_threshold=100,
                aggressive_adj=2,
                aggressive_deflection_adj=100,
                normal_deflection_adj=100,
                default_size=5,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "VOLCANIC_ROCK_VOUCHER_9500": OptionStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_9500",
                limit=self.limit["VOLCANIC_ROCK_VOUCHER_9500"],
                window=100,
                z_score_threshold=2,
                strike=9500,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "VOLCANIC_ROCK_VOUCHER_9750": OptionStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_9750",
                limit=self.limit["VOLCANIC_ROCK_VOUCHER_9750"],
                window=100,
                z_score_threshold=2,
                strike=9750,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "VOLCANIC_ROCK_VOUCHER_10000": OptionStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10000",
                limit=self.limit["VOLCANIC_ROCK_VOUCHER_10000"],
                window=100,
                z_score_threshold=2,
                strike=10000,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "VOLCANIC_ROCK_VOUCHER_10250": OptionStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10250",
                limit=self.limit["VOLCANIC_ROCK_VOUCHER_10250"],
                window=100,
                z_score_threshold=2,
                strike=10250,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "VOLCANIC_ROCK_VOUCHER_10500": OptionStrategy(
                symbol="VOLCANIC_ROCK_VOUCHER_10500",
                limit=self.limit["VOLCANIC_ROCK_VOUCHER_10500"],
                window=100,
                z_score_threshold=2,
                strike=10500,
                utils=self.utils,
                blocks=self.blocks,
            ),
            "MAGNIFICENT_MACARONS": MacaronsStrategy("MAGNIFICENT_MACARONS", self.limit["MAGNIFICENT_MACARONS"], conversion_limit=10, sunlight_threshold=50, z_score_threshold=3),
        }

    def run(self, state) -> Tuple[Dict[str, List], int, str]:
        logger.print(state.position)

        conversions = 0

        old_trader_data = jsonpickle.decode(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions
            else:
                orders[symbol] = []

            new_trader_data[symbol] = strategy.save()

        trader_data = jsonpickle.encode(new_trader_data)

        logger.flush(state, orders, conversions, trader_data)

        return orders, conversions, trader_data