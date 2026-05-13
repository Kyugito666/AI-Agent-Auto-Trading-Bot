# data/research.py
"""
Research Client — Live news & sentiment aggregation.

Sources:
  1. CryptoPanic API  : Structured crypto news feed (free tier: 1000 req/day)
  2. Playwright       : Dynamic scraping untuk sumber tanpa API
                        (alternative.me Fear & Greed, custom sources)

Arsitektur:
  - Dual-source untuk redundansi
  - NewsItem normalization: semua source → schema yang sama
  - Deduplication berdasarkan URL hash
  - Cache sederhana (TTL-based) agar tidak spam API
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
from playwright.async_api import async_playwright, Browser, BrowserContext

from config import Settings
from models.schemas import NewsItem, NewsSnapshot
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache TTL — jangan fetch news yang sama dalam 2 menit
_NEWS_CACHE_TTL = 120  # seconds


class ResearchClient:
    """
    Multi-source news aggregator dengan deduplication dan caching.
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
        self._cache: dict[str, tuple[list[NewsItem], float]] = {}  # key → (items, expiry)
        self._browser: Optional[Browser] = None
        self._playwright = None

    async def aclose(self) -> None:
        await self._http_client.aclose()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def _get_browser(self) -> Browser:
        """Lazy-init Playwright browser. Headless Chromium."""
        if self._browser is None or not self._browser.is_connected():
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            logger.info("🌐 Playwright browser initialized")
        return self._browser

    # ── Public Interface ───────────────────────────────────────────────────────

    async def fetch_cryptopanic(self, token: str) -> list[NewsItem]:
        """
        Fetch structured news dari CryptoPanic API.
        Free tier: 1000 req/day — kita cache agresif.

        Endpoint: /api/v1/posts/?auth_token=...&currencies=BTC,ETH&filter=hot
        """
        cache_key = f"cryptopanic:{token}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self.settings.CRYPTOPANIC_API_KEY:
            logger.debug("CryptoPanic API key not set — skipping")
            return []

        try:
            # Selalu sertakan BTC sebagai konteks makro
            currencies = token if token == "BTC" else f"{token},BTC"
            url = f"{self.settings.CRYPTOPANIC_BASE_URL}/posts/"
            params = {
                "auth_token": self.settings.CRYPTOPANIC_API_KEY,
                "currencies": currencies,
                "filter": "hot",          # hot | rising | bullish | bearish | important
                "public": "true",
                "kind": "news",
                "regions": "en",
            }

            response = await self._http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            items = []
            for post in data.get("results", []):
                item = self._parse_cryptopanic_post(post)
                if item:
                    items.append(item)

            # Deduplicate
            items = self._deduplicate(items)

            logger.info(f"📰 CryptoPanic: {len(items)} items for {token}")
            self._set_cache(cache_key, items)
            return items

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("CryptoPanic rate limit hit — returning empty")
            else:
                logger.warning(f"CryptoPanic HTTP error: {e}")
            return []
        except Exception as e:
            logger.warning(f"CryptoPanic fetch failed: {e}")
            return []

    async def fetch_scraped_news(self, token: str) -> list[NewsItem]:
        """
        Scrape berita dari sumber dinamis menggunakan Playwright.
        Saat ini: Fear & Greed Index + CoinDesk headlines.

        Design Note: Playwright lebih heavy dari httpx.
        Hanya digunakan jika API sources tidak cukup.
        """
        cache_key = f"scraped:{token}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        items = []

        # Fetch secara concurrent
        fear_greed_task = self._fetch_fear_greed_index()
        coindesk_task = self._scrape_coindesk_headlines(token=token)

        fear_greed_item, coindesk_items = await asyncio.gather(
            fear_greed_task,
            coindesk_task,
            return_exceptions=True,
        )

        if not isinstance(fear_greed_item, Exception) and fear_greed_item:
            items.append(fear_greed_item)
        if not isinstance(coindesk_items, Exception):
            items.extend(coindesk_items)

        items = self._deduplicate(items)
        logger.info(f"🕷️ Scraped: {len(items)} items for {token}")
        self._set_cache(cache_key, items)
        return items

    async def fetch_fear_greed_index(self) -> Optional[dict]:
        """
        Fetch Fear & Greed Index dari alternative.me.
        Tidak butuh Playwright — pure JSON API.
        Returns: {"value": 72, "classification": "Greed", "timestamp": ...}
        """
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

    # ── Private Scrapers ───────────────────────────────────────────────────────

    async def _fetch_fear_greed_index(self) -> Optional[NewsItem]:
        """Konversi Fear & Greed data menjadi NewsItem."""
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

    async def _scrape_coindesk_headlines(self, token: str) -> list[NewsItem]:
        """
        Scrape CoinDesk headlines menggunakan Playwright.
        Target: section Markets dan tagger by token.
        
        Note: Playwright digunakan karena CoinDesk menggunakan JavaScript rendering.
        """
        try:
            browser = await self._get_browser()
            context: BrowserContext = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                java_script_enabled=True,
            )
            page = await context.new_page()

            # Block media resources untuk kecepatan
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,ico}",
                lambda route: route.abort(),
            )

            url = f"https://www.coindesk.com/search?s={token}"
            await page.goto(url, wait_until="domcontentloaded", timeout=15_000)

            # Tunggu artikel muncul
            await page.wait_for_selector("article", timeout=10_000)

            articles = await page.query_selector_all("article")
            items = []

            for article in articles[:10]:  # Max 10 artikel
                try:
                    title_el = await article.query_selector("h2, h3, h4")
                    link_el = await article.query_selector("a[href]")
                    time_el = await article.query_selector("time")

                    if not title_el or not link_el:
                        continue

                    title = await title_el.inner_text()
                    href = await link_el.get_attribute("href")
                    pub_time_str = await time_el.get_attribute("datetime") if time_el else None

                    # Normalisasi URL
                    full_url = (
                        f"https://www.coindesk.com{href}"
                        if href and href.startswith("/")
                        else (href or "")
                    )

                    # Parse timestamp
                    try:
                        pub_time = (
                            datetime.fromisoformat(pub_time_str.replace("Z", "+00:00"))
                            if pub_time_str
                            else datetime.now(tz=timezone.utc)
                        )
                    except (ValueError, AttributeError):
                        pub_time = datetime.now(tz=timezone.utc)

                    item = NewsItem(
                        title=title.strip(),
                        source="CoinDesk",
                        url=full_url,
                        published_at=pub_time,
                        currencies=[token],
                        sentiment_votes={},
                    )
                    items.append(item)

                except Exception as e:
                    logger.debug(f"Error parsing CoinDesk article: {e}")
                    continue

            await context.close()
            logger.debug(f"CoinDesk scraped: {len(items)} articles for {token}")
            return items

        except Exception as e:
            logger.warning(f"CoinDesk scrape failed: {e}")
            return []

    # ── Utility Methods ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_cryptopanic_post(post: dict) -> Optional[NewsItem]:
        """Map CryptoPanic API response → NewsItem schema."""
        try:
            title = post.get("title", "").strip()
            url = post.get("url", "")
            source_info = post.get("source", {})
            source = source_info.get("title", "Unknown")

            published_at_str = post.get("published_at", "")
            try:
                published_at = datetime.fromisoformat(
                    published_at_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                published_at = datetime.now(tz=timezone.utc)

            # Extract currencies
            currencies = [
                c.get("code", "")
                for c in post.get("currencies", [])
                if c.get("code")
            ]

            # Sentiment votes dari CryptoPanic
            votes = post.get("votes", {})
            sentiment_votes = {
                "bullish": votes.get("positive", 0),
                "bearish": votes.get("negative", 0),
                "important": votes.get("important", 0),
                "lol": votes.get("lol", 0),
            }

            if not title or not url:
                return None

            return NewsItem(
                title=title,
                source=source,
                url=url,
                published_at=published_at,
                currencies=currencies,
                sentiment_votes=sentiment_votes,
            )
        except Exception:
            return None

    @staticmethod
    def _deduplicate(items: list[NewsItem]) -> list[NewsItem]:
        """Remove duplikat berdasarkan URL hash."""
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