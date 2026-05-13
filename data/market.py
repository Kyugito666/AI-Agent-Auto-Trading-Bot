# data/market.py
"""
Market Data Client — OHLCV, ticker, dan order book.

Source: Binance Public REST API (tidak butuh API key).
Endpoint publik, rate limit generous: 1200 req/menit (weight-based).

Design:
  - Semua method async + retry
  - Response di-parse ke dataclass yang type-safe
  - Cache ticker 5 detik (hindari spam pada pipeline yang berjalan cepat)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache TTL untuk ticker (harga bergerak cepat, tapi 5s cukup untuk menghindari spam)
_TICKER_CACHE_TTL = 5.0  # seconds


@dataclass
class Ticker:
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_usdt: float        # Quote volume (USDT)
    volume_base: float        # Base volume (e.g., BTC)
    change_24h_pct: float
    high_24h: float
    low_24h: float
    timestamp: datetime


@dataclass
class Kline:
    """Satu candlestick OHLCV."""
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float       # Volume dalam USDT
    trades: int               # Jumlah trades dalam periode ini


@dataclass
class OrderBookLevel:
    price: float
    quantity: float


@dataclass
class OrderBook:
    symbol: str
    bids: list[OrderBookLevel]   # Sorted: highest bid dulu
    asks: list[OrderBookLevel]   # Sorted: lowest ask dulu
    timestamp: datetime

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def spread_pct(self) -> float:
        if self.bids and self.bids[0].price > 0:
            return (self.spread / self.bids[0].price) * 100
        return 0.0

    @property
    def bid_depth(self) -> float:
        """Total USDT value di sisi bid (5 level)."""
        return sum(b.price * b.quantity for b in self.bids[:5])

    @property
    def ask_depth(self) -> float:
        """Total USDT value di sisi ask (5 level)."""
        return sum(a.price * a.quantity for a in self.asks[:5])

    @property
    def imbalance_ratio(self) -> float:
        """
        Order book imbalance: > 1.0 → lebih banyak bids (bullish pressure).
        Formula: bid_depth / (bid_depth + ask_depth)
        """
        total = self.bid_depth + self.ask_depth
        return self.bid_depth / total if total > 0 else 0.5


class MarketDataClient:
    """
    Binance public API client.
    Zero credentials required — hanya public endpoints.
    """

    # Binance kline interval yang valid
    VALID_INTERVALS = {
        "1m", "3m", "5m", "15m", "30m",
        "1h", "2h", "4h", "6h", "8h", "12h",
        "1d", "3d", "1w", "1M",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self._base_url = settings.BINANCE_BASE_URL
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            headers={"Content-Type": "application/json"},
        )
        # In-memory ticker cache: symbol → (Ticker, expiry_timestamp)
        self._ticker_cache: dict[str, tuple[Ticker, float]] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Fetch 24hr ticker statistics.
        Cached 5 detik untuk menghindari redundant calls dalam satu pipeline cycle.
        """
        sym = symbol.upper()

        # Check cache
        if sym in self._ticker_cache:
            cached_ticker, expiry = self._ticker_cache[sym]
            if time.monotonic() < expiry:
                return cached_ticker

        data = await self._fetch("/api/v3/ticker/24hr", params={"symbol": sym})

        ticker = Ticker(
            symbol=sym,
            last_price=float(data["lastPrice"]),
            bid=float(data["bidPrice"]),
            ask=float(data["askPrice"]),
            volume_usdt=float(data["quoteVolume"]),
            volume_base=float(data["volume"]),
            change_24h_pct=float(data["priceChangePercent"]),
            high_24h=float(data["highPrice"]),
            low_24h=float(data["lowPrice"]),
            timestamp=datetime.now(tz=timezone.utc),
        )

        self._ticker_cache[sym] = (ticker, time.monotonic() + _TICKER_CACHE_TTL)
        logger.debug(
            f"Ticker {sym}: ${ticker.last_price:,.4f} | "
            f"24h: {ticker.change_24h_pct:+.2f}%"
        )
        return ticker

    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[Kline]:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol   : Trading pair (e.g., "BTCUSDT")
            interval : Candlestick interval (e.g., "1h", "4h", "1d")
            limit    : Jumlah candles (max 1000 per request)
            start_time/end_time: Unix timestamp miliseconds (optional)
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {self.VALID_INTERVALS}")

        limit = min(limit, 1000)  # Binance hard limit
        sym = symbol.upper()

        params: dict = {"symbol": sym, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        raw = await self._fetch("/api/v3/klines", params=params)

        klines = [
            Kline(
                open_time=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                close_time=datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
                quote_volume=float(k[7]),
                trades=int(k[8]),
            )
            for k in raw
        ]

        logger.debug(f"Klines {sym} {interval}: {len(klines)} candles fetched")
        return klines

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """
        Fetch order book snapshot.
        Berguna untuk menilai buying/selling pressure (imbalance ratio).

        Args:
            depth: Jumlah level (5, 10, 20, 50, 100, 500, 1000)
        """
        sym = symbol.upper()
        valid_depths = {5, 10, 20, 50, 100, 500, 1000}
        depth = min((d for d in valid_depths if d >= depth), default=20)

        data = await self._fetch("/api/v3/depth", params={"symbol": sym, "limit": depth})

        bids = [
            OrderBookLevel(price=float(b[0]), quantity=float(b[1]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), quantity=float(a[1]))
            for a in data.get("asks", [])
        ]

        ob = OrderBook(
            symbol=sym,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(tz=timezone.utc),
        )

        logger.debug(
            f"OrderBook {sym}: spread={ob.spread_pct:.4f}% | "
            f"imbalance={ob.imbalance_ratio:.2f}"
        )
        return ob

    async def get_current_price(self, symbol: str) -> float:
        """Lightweight price fetch — hanya harga terkini, tanpa extra data."""
        sym = symbol.upper()
        data = await self._fetch("/api/v3/ticker/price", params={"symbol": sym})
        return float(data["price"])

    async def get_multi_ticker(self, symbols: list[str]) -> dict[str, Ticker]:
        """
        Fetch ticker untuk multiple symbols secara parallel.
        Lebih efisien daripada sequential calls.
        """
        import asyncio
        tasks = [self.get_ticker(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            sym: result
            for sym, result in zip(symbols, results)
            if not isinstance(result, Exception)
        }

    def compute_technical_indicators(self, klines: list[Kline]) -> dict:
        """
        Hitung indikator teknikal dasar dari klines.
        Pure computation — tidak ada I/O.

        Returns dict dengan: ema_7, ema_21, rsi_14, atr_14, vwap
        """
        if len(klines) < 21:
            return {}

        closes = [k.close for k in klines]
        highs = [k.high for k in klines]
        lows = [k.low for k in klines]
        volumes = [k.volume for k in klines]

        return {
            "ema_7": self._ema(closes, 7),
            "ema_21": self._ema(closes, 21),
            "rsi_14": self._rsi(closes, 14),
            "atr_14": self._atr(highs, lows, closes, 14),
            "vwap": self._vwap(klines),
            "current_price": closes[-1],
        }

    # ── Technical Indicator Helpers ────────────────────────────────────────────

    @staticmethod
    def _ema(prices: list[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        k = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * k + ema * (1 - k)
        return round(ema, 8)

    @staticmethod
    def _rsi(closes: list[float], period: int = 14) -> float:
        """Relative Strength Index."""
        if len(closes) < period + 1:
            return 50.0

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d for d in deltas if d > 0]
        losses = [-d for d in deltas if d < 0]

        if not gains:
            return 0.0
        if not losses:
            return 100.0

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)

    @staticmethod
    def _atr(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float:
        """Average True Range — ukuran volatilitas."""
        if len(closes) < 2:
            return 0.0

        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        atr_vals = true_ranges[-period:]
        return round(sum(atr_vals) / len(atr_vals), 8) if atr_vals else 0.0

    @staticmethod
    def _vwap(klines: list[Kline]) -> float:
        """
        Volume Weighted Average Price.
        Harga rata-rata berbobot volume — referensi institusional.
        """
        numerator = sum(
            ((k.high + k.low + k.close) / 3) * k.volume
            for k in klines
        )
        denominator = sum(k.volume for k in klines)
        return round(numerator / denominator, 8) if denominator > 0 else 0.0

    # ── HTTP Helper ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _fetch(self, path: str, params: dict | None = None) -> dict | list:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()