# models/schemas.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class BTCMarketState:
    price: float
    change_24h_pct: float
    trend: str              # "BULLISH" | "BEARISH" | "SIDEWAYS"
    volume_24h_usdt: float
    dominance_pct: float
    timestamp: datetime


@dataclass
class WhaleTransaction:
    tx_hash: str
    amount_usd: float
    direction: str          # "IN" | "OUT"
    from_address: str
    to_address: str
    timestamp: datetime


@dataclass
class OnChainSnapshot:
    token: str
    whale_transactions: list[WhaleTransaction]
    dex_liquidity_usd: float
    volume_anomaly_score: float
    large_transfers: list[WhaleTransaction]
    timestamp: datetime

    @classmethod
    def empty(cls) -> "OnChainSnapshot":
        return cls(
            token="UNKNOWN", whale_transactions=[], dex_liquidity_usd=0.0,
            volume_anomaly_score=0.0, large_transfers=[],
            timestamp=datetime.utcnow(),
        )


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: datetime
    currencies: list[str] = field(default_factory=list)
    sentiment_votes: dict = field(default_factory=dict)


@dataclass
class NewsSnapshot:
    token: str
    items: list[NewsItem]
    timestamp: datetime

    @classmethod
    def empty(cls) -> "NewsSnapshot":
        return cls(token="UNKNOWN", items=[], timestamp=datetime.utcnow())


@dataclass
class FullAnalysisContext:
    symbol: str
    timestamp: datetime
    btc_state: Optional[BTCMarketState]
    onchain: OnChainSnapshot
    news: NewsSnapshot
    sentiment_score: float