"""
Binance Futures strategy based on:
- order book imbalance
- trade delta (aggressive buy vs sell volume)
- volume filter

This script is intentionally conservative:
- starts in paper mode by default
- uses simple REST polling for clarity
- supports Binance USDT-M Futures

Install:
    pip install python-binance pandas numpy

Environment variables:
    BINANCE_API_KEY
    BINANCE_API_SECRET

Example:
    python binance_orderbook_delta_strategy.py --symbol BTCUSDT --paper
    python binance_orderbook_delta_strategy.py --symbol BTCUSDT --live
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client


@dataclass
class StrategyConfig:
    symbol: str = "BTCUSDT"
    interval: str = Client.KLINE_INTERVAL_1MINUTE
    depth_levels: int = 20
    trade_lookback_limit: int = 500
    volume_ma_period: int = 20
    ema_period: int = 50
    atr_period: int = 14
    loop_sleep_sec: int = 10
    orderbook_imbalance_threshold: float = 0.12
    delta_threshold: float = 0.15
    min_volume_ratio: float = 1.2
    risk_per_trade: float = 0.01
    leverage: int = 5
    qty_precision: int = 3
    paper: bool = True
    use_isolated_margin: bool = True


class BinanceOrderBookDeltaStrategy:
    def __init__(self, client: Client, cfg: StrategyConfig):
        self.client = client
        self.cfg = cfg
        self.paper_position_amt = 0.0
        self.paper_entry_price = 0.0
        self.paper_balance = 1000.0
        self.qty_step = 0.001
        self.min_qty = 0.0
        self._setup_symbol()

    def _setup_symbol(self) -> None:
        if self.cfg.paper:
            return

        try:
            info = self.client.futures_exchange_info()
            symbol_info = next(
                item for item in info["symbols"] if item["symbol"] == self.cfg.symbol
            )
            lot_filter = next(
                f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"
            )
            self.qty_step = float(lot_filter["stepSize"])
            self.min_qty = float(lot_filter["minQty"])
        except Exception:
            pass

        try:
            self.client.futures_change_margin_type(
                symbol=self.cfg.symbol,
                marginType="ISOLATED" if self.cfg.use_isolated_margin else "CROSSED",
            )
        except Exception:
            pass

        try:
            self.client.futures_change_leverage(
                symbol=self.cfg.symbol,
                leverage=self.cfg.leverage,
            )
        except Exception:
            pass

    def get_klines(self, limit: int = 200) -> pd.DataFrame:
        rows = self.client.futures_klines(
            symbol=self.cfg.symbol,
            interval=self.cfg.interval,
            limit=limit,
        )
        df = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_volume",
                "taker_buy_quote_volume",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume", "taker_buy_base_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_orderbook_imbalance(self) -> Tuple[float, float, float]:
        depth = self.client.futures_order_book(
            symbol=self.cfg.symbol,
            limit=self.cfg.depth_levels,
        )
        bids = np.array([[float(p), float(q)] for p, q in depth["bids"]], dtype=float)
        asks = np.array([[float(p), float(q)] for p, q in depth["asks"]], dtype=float)

        bid_notional = float(np.sum(bids[:, 0] * bids[:, 1])) if len(bids) else 0.0
        ask_notional = float(np.sum(asks[:, 0] * asks[:, 1])) if len(asks) else 0.0
        total = bid_notional + ask_notional
        imbalance = 0.0 if total == 0 else (bid_notional - ask_notional) / total
        return imbalance, bid_notional, ask_notional

    def get_trade_delta(self) -> Tuple[float, float, float]:
        trades = self.client.futures_aggregate_trades(
            symbol=self.cfg.symbol,
            limit=self.cfg.trade_lookback_limit,
        )
        buy_volume = 0.0
        sell_volume = 0.0
        for t in trades:
            qty = float(t["q"])
            is_buyer_maker = bool(t["m"])
            if is_buyer_maker:
                sell_volume += qty
            else:
                buy_volume += qty
        total = buy_volume + sell_volume
        delta = 0.0 if total == 0 else (buy_volume - sell_volume) / total
        return delta, buy_volume, sell_volume

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean().iloc[-1]
        return float(atr)

    def compute_signal(self) -> dict:
        df = self.get_klines(limit=max(200, self.cfg.ema_period + 20))
        last_close = float(df["close"].iloc[-1])
        ema = float(df["close"].ewm(span=self.cfg.ema_period, adjust=False).mean().iloc[-1])
        volume_ma = float(df["volume"].rolling(self.cfg.volume_ma_period).mean().iloc[-1])
        last_volume = float(df["volume"].iloc[-1])
        volume_ratio = 0.0 if volume_ma == 0 else last_volume / volume_ma
        atr = self._atr(df, self.cfg.atr_period)
        orderbook_imbalance, bid_notional, ask_notional = self.get_orderbook_imbalance()
        delta, buy_volume, sell_volume = self.get_trade_delta()

        trend_up = last_close > ema
        trend_down = last_close < ema
        volume_ok = volume_ratio >= self.cfg.min_volume_ratio

        long_score = 0
        short_score = 0

        if orderbook_imbalance >= self.cfg.orderbook_imbalance_threshold:
            long_score += 1
        if orderbook_imbalance <= -self.cfg.orderbook_imbalance_threshold:
            short_score += 1

        if delta >= self.cfg.delta_threshold:
            long_score += 1
        if delta <= -self.cfg.delta_threshold:
            short_score += 1

        if trend_up:
            long_score += 1
        if trend_down:
            short_score += 1

        if volume_ok and last_volume > 0:
            if delta > 0:
                long_score += 1
            elif delta < 0:
                short_score += 1

        side: Optional[str] = None
        if long_score >= 3 and long_score > short_score:
            side = "BUY"
        elif short_score >= 3 and short_score > long_score:
            side = "SELL"

        return {
            "side": side,
            "last_close": last_close,
            "ema": ema,
            "volume_ratio": volume_ratio,
            "atr": atr,
            "orderbook_imbalance": orderbook_imbalance,
            "bid_notional": bid_notional,
            "ask_notional": ask_notional,
            "delta": delta,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "long_score": long_score,
            "short_score": short_score,
        }

    def get_position(self) -> dict:
        if self.cfg.paper:
            return {
                "positionAmt": self.paper_position_amt,
                "entryPrice": self.paper_entry_price,
            }

        positions = self.client.futures_position_information(symbol=self.cfg.symbol)
        if not positions:
            return {"positionAmt": 0.0, "entryPrice": 0.0}
        pos = positions[0]
        return {
            "positionAmt": float(pos["positionAmt"]),
            "entryPrice": float(pos["entryPrice"]),
        }

    def _round_qty(self, qty: float) -> float:
        if self.cfg.paper:
            precision = max(0, self.cfg.qty_precision)
            return float(f"{qty:.{precision}f}")

        if self.qty_step <= 0:
            precision = max(0, self.cfg.qty_precision)
            return float(f"{qty:.{precision}f}")

        steps = np.floor(qty / self.qty_step)
        rounded = steps * self.qty_step
        if self.min_qty > 0 and rounded < self.min_qty:
            rounded = self.min_qty
        return float(f"{rounded:.8f}")

    def estimate_order_qty(self, price: float, atr: float) -> float:
        if self.cfg.paper:
            balance = self.paper_balance
        else:
            balances = self.client.futures_account_balance()
            usdt_balance = next(
                (float(item["balance"]) for item in balances if item.get("asset") == "USDT"),
                float(balances[0]["balance"]) if balances else 0.0,
            )
            balance = usdt_balance

        risk_usdt = balance * self.cfg.risk_per_trade
        stop_distance = max(atr * 1.5, price * 0.003)
        raw_qty = (risk_usdt * self.cfg.leverage) / stop_distance
        return self._round_qty(raw_qty)

    def place_entry(self, side: str, qty: float, price: Optional[float] = None) -> Optional[dict]:
        if qty <= 0:
            return None

        if self.cfg.paper:
            if price is None:
                price = 0.0
            self.paper_position_amt = qty if side == "BUY" else -qty
            self.paper_entry_price = price
            return {"paper": True, "side": side, "qty": qty}

        return self.client.futures_create_order(
            symbol=self.cfg.symbol,
            side=side,
            type="MARKET",
            quantity=qty,
        )

    def place_exit(self, side: str, qty: float, price: Optional[float] = None, reduce_only: bool = True) -> Optional[dict]:
        if qty <= 0:
            return None

        if self.cfg.paper:
            if price is None:
                price = 0.0
            self.paper_position_amt = 0.0
            self.paper_entry_price = 0.0
            return {"paper": True, "exit_side": side, "qty": qty, "reduceOnly": reduce_only}

        return self.client.futures_create_order(
            symbol=self.cfg.symbol,
            side=side,
            type="MARKET",
            quantity=qty,
            reduceOnly=reduce_only,
        )

    def manage_position(self, signal: dict) -> None:
        position = self.get_position()
        position_amt = position["positionAmt"]
        side = signal["side"]
        last_close = signal["last_close"]
        atr = signal["atr"]

        if side is None:
            return

        if position_amt == 0:
            qty = self.estimate_order_qty(last_close, atr)
            self.place_entry(side, qty, price=last_close)
            print(f"ENTRY {side} qty={qty} price={last_close:.2f}")
            return

        # Flip only if current position disagrees strongly with the signal.
        if position_amt > 0 and side == "SELL":
            self.place_exit("SELL", abs(position_amt), price=last_close)
            qty = self.estimate_order_qty(last_close, atr)
            self.place_entry("SELL", qty, price=last_close)
            print(f"FLIP to SELL qty={qty} price={last_close:.2f}")
        elif position_amt < 0 and side == "BUY":
            self.place_exit("BUY", abs(position_amt), price=last_close)
            qty = self.estimate_order_qty(last_close, atr)
            self.place_entry("BUY", qty, price=last_close)
            print(f"FLIP to BUY qty={qty} price={last_close:.2f}")

    def run_once(self) -> dict:
        signal = self.compute_signal()
        print(
            "[{symbol}] side={side} close={close:.2f} ema={ema:.2f} "
            "imb={imb:.3f} delta={delta:.3f} vol_ratio={vol:.2f} "
            "LS={ls} SS={ss}".format(
                symbol=self.cfg.symbol,
                side=signal["side"],
                close=signal["last_close"],
                ema=signal["ema"],
                imb=signal["orderbook_imbalance"],
                delta=signal["delta"],
                vol=signal["volume_ratio"],
                ls=signal["long_score"],
                ss=signal["short_score"],
            )
        )
        self.manage_position(signal)
        return signal

    def run_forever(self) -> None:
        print(
            f"Starting strategy for {self.cfg.symbol} "
            f"mode={'PAPER' if self.cfg.paper else 'LIVE'}"
        )
        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"ERROR: {exc}")
            time.sleep(self.cfg.loop_sleep_sec)


def build_client() -> Client:
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    return Client(api_key, api_secret)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance Futures order book + delta strategy")
    parser.add_argument("--symbol", default="BTCUSDT", help="Futures symbol, e.g. BTCUSDT")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--paper", action="store_true", help="Force paper trading")
    parser.add_argument("--leverage", type=int, default=5, help="Leverage")
    parser.add_argument("--sleep", type=int, default=10, help="Loop sleep seconds")
    parser.add_argument("--levels", type=int, default=20, help="Order book depth levels")
    parser.add_argument("--delta-threshold", type=float, default=0.15, help="Delta threshold")
    parser.add_argument("--imbalance-threshold", type=float, default=0.12, help="Order book imbalance threshold")
    parser.add_argument("--volume-ratio", type=float, default=1.2, help="Min volume / volume MA ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper = True
    if args.live:
        paper = False
    if args.paper:
        paper = True

    cfg = StrategyConfig(
        symbol=args.symbol.upper(),
        paper=paper,
        leverage=args.leverage,
        loop_sleep_sec=args.sleep,
        depth_levels=args.levels,
        delta_threshold=args.delta_threshold,
        orderbook_imbalance_threshold=args.imbalance_threshold,
        min_volume_ratio=args.volume_ratio,
    )

    if not paper and (not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET")):
        raise SystemExit("For --live you must set BINANCE_API_KEY and BINANCE_API_SECRET")

    client = build_client()
    strategy = BinanceOrderBookDeltaStrategy(client, cfg)
    strategy.run_forever()


if __name__ == "__main__":
    main()
