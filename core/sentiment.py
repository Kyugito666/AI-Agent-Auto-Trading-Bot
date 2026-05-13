# core/sentiment.py
"""
Sentiment Scorer — Mengonversi raw news items → skor numerik -100 hingga +100.

Pipeline:
  1. Keyword scoring  (cepat, deterministic)
  2. Vote aggregation (dari CryptoPanic votes)
  3. Fear & Greed integration (dari alternative.me)
  4. LLM micro-scoring (sampling 3-5 headline paling penting)

Final score = weighted average dari semua komponen.

Design: Tidak ada external I/O di class ini. Data harus sudah ada.
        LLM call yang ada di sini menggunakan endpoint yang sama dengan LLMEngine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import httpx

from config import get_settings
from models.schemas import NewsItem
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)

# ── Keyword Dictionaries ───────────────────────────────────────────────────────
# Dibangun berdasarkan domain crypto — lebih akurat dari general sentiment lexicon

BULLISH_KEYWORDS: dict[str, float] = {
    # Strong signals (weight 3.0)
    "breakout": 3.0, "all-time high": 3.0, "ath": 3.0, "surge": 3.0,
    "institutional adoption": 3.0, "etf approved": 3.0, "halving": 2.5,
    # Medium signals (weight 2.0)
    "bull run": 2.0, "accumulation": 2.0, "oversold": 2.0, "bottom": 2.0,
    "partnership": 2.0, "listing": 2.0, "upgrade": 2.0, "milestone": 2.0,
    "rally": 2.0, "recovery": 2.0, "moon": 1.5, "pump": 1.5,
    # Weak signals (weight 1.0)
    "bullish": 1.0, "positive": 1.0, "growth": 1.0, "gain": 1.0,
    "rise": 1.0, "increase": 1.0, "support": 1.0, "demand": 1.0,
}

BEARISH_KEYWORDS: dict[str, float] = {
    # Strong signals (weight 3.0)
    "hack": 3.0, "exploit": 3.0, "rug pull": 3.0, "scam": 3.0,
    "sec lawsuit": 3.0, "ban": 3.0, "crash": 3.0, "bankrupt": 3.0,
    "liquidation": 2.8, "insolvency": 2.8,
    # Medium signals (weight 2.0)
    "bear market": 2.0, "overbought": 2.0, "dump": 2.0, "sell-off": 2.0,
    "regulation": 1.5, "crackdown": 2.0, "fine": 1.5, "fraud": 2.5,
    "warning": 1.5, "risk": 1.0, "concern": 1.0, "investigate": 2.0,
    # Weak signals (weight 1.0)
    "bearish": 1.0, "negative": 1.0, "decline": 1.0, "drop": 1.0,
    "fall": 1.0, "decrease": 1.0, "resistance": 1.0, "sell": 0.8,
}

# Kata yang menginversi sentiment jika muncul sebelum keyword
NEGATION_WORDS = {"not", "no", "never", "without", "despite", "against", "avoid"}


@dataclass
class SentimentComponents:
    """Komponen individual sebelum diaggregasi."""
    keyword_score: float        # -100 to +100
    vote_score: float           # -100 to +100 (dari CryptoPanic votes)
    fear_greed_score: float     # -100 to +100 (remapped dari 0-100)
    llm_micro_score: float      # -100 to +100 (LLM rating beberapa headline)
    news_count: int
    dominant_keywords: list[str]


class SentimentScorer:
    """
    Multi-component sentiment aggregator.
    Skor akhir: weighted average dari semua komponen.
    """

    # Bobot setiap komponen
    WEIGHTS = {
        "keyword": 0.30,
        "vote": 0.25,
        "fear_greed": 0.20,
        "llm_micro": 0.25,
    }

    def __init__(self):
        self.settings = get_settings()
        self._http_client = httpx.AsyncClient(timeout=15.0)
        self._rate_limiter = RateLimiter(max_calls=10, period_seconds=60)
        self._fear_greed_cache: Optional[tuple[float, float]] = None  # (score, expiry)

    async def compute_score(self, news_items: list[NewsItem]) -> float:
        """
        Main entry point. Hitung skor sentimen dari list NewsItem.

        Returns:
            float: Skor -100 (extreme fear/bearish) hingga +100 (extreme greed/bullish)
        """
        if not news_items:
            logger.debug("No news items — returning neutral sentiment 0.0")
            return 0.0

        # Compute semua komponen
        keyword_score = self._compute_keyword_score(news_items)
        vote_score = self._compute_vote_score(news_items)
        fear_greed = await self._get_fear_greed_score()
        llm_score = await self._compute_llm_micro_score(news_items[:5])

        components = SentimentComponents(
            keyword_score=keyword_score,
            vote_score=vote_score,
            fear_greed_score=fear_greed,
            llm_micro_score=llm_score,
            news_count=len(news_items),
            dominant_keywords=self._extract_dominant_keywords(news_items),
        )

        # Weighted average
        final_score = (
            components.keyword_score * self.WEIGHTS["keyword"]
            + components.vote_score * self.WEIGHTS["vote"]
            + components.fear_greed_score * self.WEIGHTS["fear_greed"]
            + components.llm_micro_score * self.WEIGHTS["llm_micro"]
        )

        final_score = max(-100.0, min(100.0, final_score))

        logger.info(
            f"📊 Sentiment | final={final_score:+.1f} | "
            f"keyword={keyword_score:+.1f} | vote={vote_score:+.1f} | "
            f"f&g={fear_greed:+.1f} | llm={llm_score:+.1f} | "
            f"news_count={len(news_items)}"
        )
        return round(final_score, 2)

    # ── Component Scorers ──────────────────────────────────────────────────────

    def _compute_keyword_score(self, items: list[NewsItem]) -> float:
        """
        NLP keyword matching dengan negation handling.
        Aggregate score dari semua headlines.
        """
        total_bullish = 0.0
        total_bearish = 0.0

        for item in items:
            text = item.title.lower()
            words = re.findall(r'\b\w+\b', text)
            word_set = set(words)

            for i, word in enumerate(words):
                # Check negation (kata sebelumnya)
                negated = i > 0 and words[i - 1] in NEGATION_WORDS

                # Multi-word phrases (2-gram)
                bigram = f"{words[i-1]} {word}" if i > 0 else ""

                for keyword, weight in BULLISH_KEYWORDS.items():
                    if keyword in text and (keyword == word or keyword == bigram):
                        if negated:
                            total_bearish += weight * 0.5  # Negated bullish = slight bearish
                        else:
                            total_bullish += weight

                for keyword, weight in BEARISH_KEYWORDS.items():
                    if keyword in text and (keyword == word or keyword == bigram):
                        if negated:
                            total_bullish += weight * 0.5  # Negated bearish = slight bullish
                        else:
                            total_bearish += weight

        if total_bullish == 0 and total_bearish == 0:
            return 0.0

        # Normalize ke -100..+100
        net = total_bullish - total_bearish
        total = total_bullish + total_bearish
        score = (net / total) * 100

        return round(score, 2)

    def _compute_vote_score(self, items: list[NewsItem]) -> float:
        """
        Aggregate sentiment dari CryptoPanic community votes.
        Votes: bullish, bearish, important, lol
        """
        total_bull = 0
        total_bear = 0

        for item in items:
            votes = item.sentiment_votes
            total_bull += votes.get("bullish", 0) + votes.get("positive", 0)
            total_bear += votes.get("bearish", 0) + votes.get("negative", 0)

        total = total_bull + total_bear
        if total == 0:
            return 0.0

        score = ((total_bull - total_bear) / total) * 100
        return round(score, 2)

    async def _get_fear_greed_score(self) -> float:
        """
        Fear & Greed Index dari alternative.me.
        Di-cache 10 menit karena index tidak berubah sering.

        Raw value 0-100 → remapped ke -100..+100:
          0   → -100 (extreme fear)
          50  → 0    (neutral)
          100 → +100 (extreme greed)
        """
        import time
        now = time.monotonic()

        if self._fear_greed_cache:
            score, expiry = self._fear_greed_cache
            if now < expiry:
                return score

        try:
            response = await self._http_client.get(
                "https://api.alternative.me/fng/?limit=1&format=json",
                timeout=8.0,
            )
            response.raise_for_status()
            data = response.json()
            raw_value = int(data["data"][0]["value"])

            # Remap 0-100 → -100..+100
            score = (raw_value - 50) * 2
            self._fear_greed_cache = (float(score), now + 600)  # Cache 10 menit
            return float(score)

        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e} — using 0")
            return 0.0

    async def _compute_llm_micro_score(self, items: list[NewsItem]) -> float:
        """
        Minta LLM untuk rating sentimen beberapa headline paling penting.
        Menggunakan Groq langsung dengan prompt minimal untuk efisiensi token.

        Hanya dijalankan jika ada headline yang cukup.
        """
        if not items or not self.settings.GROQ_API_KEY:
            return 0.0

        headlines = "\n".join(
            f"{i+1}. [{item.source}] {item.title}"
            for i, item in enumerate(items[:5])
        )

        prompt = (
            "Rate the overall crypto market sentiment from these headlines.\n"
            "Respond with ONLY a JSON object: {\"score\": <integer from -100 to 100>}\n"
            "Where -100=extremely bearish, 0=neutral, 100=extremely bullish.\n\n"
            f"Headlines:\n{headlines}"
        )

        try:
            async with self._rate_limiter:
                response = await self._http_client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.settings.GROQ_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 30,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=10.0,
                )
                response.raise_for_status()

                import json
                content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                score = float(parsed.get("score", 0))
                return max(-100.0, min(100.0, score))

        except Exception as e:
            logger.warning(f"LLM micro-score failed: {e} — returning 0")
            return 0.0

    def _extract_dominant_keywords(self, items: list[NewsItem]) -> list[str]:
        """Temukan keyword yang paling sering muncul di semua headlines."""
        freq: dict[str, int] = {}
        all_text = " ".join(item.title.lower() for item in items)

        for kw in {**BULLISH_KEYWORDS, **BEARISH_KEYWORDS}:
            count = all_text.count(kw)
            if count > 0:
                freq[kw] = count

        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:5]]