# data/research.py
"""
Research Client — Live news & sentiment aggregation.

Sources:
  1. CryptoCompare API : Structured crypto news feed (Free, no key needed)
  2. Alternative.me    : Fear & Greed Index
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Optional

import httpx

from config import Settings
from models.schemas import NewsItem, NewsSnapshot
from utils.logger import get_logger

logger = get_logger(__name__)

_NEWS_CACHE_TTL = 120  # seconds


class ResearchClient:
    """
    Multi-source news aggregator dengan deduplication dan caching.
    (Playwright dihapus agar stabil di Windows & hemat memori)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; TradingAgent/1.0)",
                "Accept": "application/json",
            },
        )
        self._cache: dict[str, tuple[list[NewsItem], float]] = {}

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def fetch_cryptopanic(self, token: str) -> list[NewsItem]:
        """
        Fetch dari CryptoCompare (Pengganti CryptoPanic).
        """
        cache_key = f"cryptocompare:{token}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            params = {
                "lang": "EN",
                "categories": token
            }

            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            items = []
            for post in data.get("Data", []):
                try:
                    published_at = datetime.fromtimestamp(
                        post.get("published_on", 0), tz=timezone.utc
                    )
                    item = NewsItem(
                        title=post.get("title", "").strip(),
                        source=post.get("source_info", {}).get("name", "Unknown"),
                        url=post.get("url", ""),
                        published_at=published_at,
                        currencies=[token],
                        sentiment_votes={}, 
                    )
                    items.append(item)
                except Exception as e:
                    logger.debug(f"Error parsing CryptoCompare article: {e}")
                    continue

            items = self._deduplicate(items)
            logger.info(f"📰 CryptoCompare: {len(items)} items for {token}")
            self._set_cache(cache_key, items)
            return items

        except Exception as e:
            logger.warning(f"CryptoCompare fetch failed: {e}")
            return []

    async def fetch_scraped_news(self, token: str) -> list[NewsItem]:
        """
        Fetch Fear & Greed Index aja.
        """
        cache_key = f"scraped:{token}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        items = []
        fear_greed_item = await self._fetch_fear_greed_index_item()
        
        if fear_greed_item:
            items.append(fear_greed_item)

        logger.info(f"🕷️ Scraped (F&G only): {len(items)} items for {token}")
        self._set_cache(cache_key, items)
        return items

    async def fetch_fear_greed_index(self) -> Optional[dict]:
        try:
            response = await self._http_client.get(
                "https://api.alternative.me/fng/?limit=2&format=json"
            )
            response.raise_for_status()
            data = response.json()

            items = data.get("data", [])
            if not items:
                return None

            latest = items[0]
            return {
                "value": int(latest.get("value", 50)),
                "classification": latest.get("value_classification", "Neutral"),
                "timestamp": latest.get("timestamp"),
            }
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return None

    async def _fetch_fear_greed_index_item(self) -> Optional[NewsItem]:
        data = await self.fetch_fear_greed_index()
        if not data:
            return None

        value = data["value"]
        classification = data["classification"]

        return NewsItem(
            title=f"Crypto Fear & Greed Index: {value}/100 — {classification}",
            source="alternative.me",
            url="https://alternative.me/crypto/fear-and-greed-index/",
            published_at=datetime.now(tz=timezone.utc),
            currencies=["BTC", "ETH"],
            sentiment_votes={
                "bullish": max(0, value - 50),
                "bearish": max(0, 50 - value),
            },
        )

    @staticmethod
    def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
        seen: set[str] = set()
        unique = []
        for item in items:
            url_hash = hashlib.md5(item.url.encode()).hexdigest()
            if url_hash not in seen:
                seen.add(url_hash)
                unique.append(item)
        return unique

    def _get_cached(self, key: str) -> Optional[list[NewsItem]]:
        if key in self._cache:
            items, expiry = self._cache[key]
            if expiry > datetime.now(tz=timezone.utc).timestamp():
                return items
            del self._cache[key]
        return None

    def _set_cache(self, key: str, items: list[NewsItem]) -> None:
        expiry = datetime.now(tz=timezone.utc).timestamp() + _NEWS_CACHE_TTL
        self._cache[key] = (items, expiry)
