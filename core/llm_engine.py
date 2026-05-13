# core/llm_engine.py
"""
LLM Engine dengan strategi Groq-primary / Gemini-fallback.
Menggunakan structured prompt engineering untuk output yang deterministik.
"""
import json
from dataclasses import dataclass
from typing import Optional

import httpx

from config import Settings
from models.schemas import FullAnalysisContext
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter

logger = get_logger(__name__)

ANALYSIS_SYSTEM_PROMPT = """You are an elite quantitative crypto trading analyst with deep expertise in:
- On-chain data interpretation (whale movements, liquidity analysis)
- Macro-economic impact on crypto markets
- Technical analysis (trend, momentum, volume)
- Market sentiment analysis

Your role is to synthesize all provided data and produce a structured trading analysis.

CRITICAL RULES:
1. Always respond with VALID JSON only — no markdown, no explanation outside JSON
2. Be precise and data-driven, not speculative
3. Confidence score must reflect data quality and signal convergence
4. Risk assessment must consider BTC correlation if analyzing altcoins

Required JSON structure:
{
  "market_summary": "2-3 sentence summary of current market state",
  "btc_impact": "How BTC trend impacts this asset (or 'N/A' if analyzing BTC)",
  "onchain_insight": "Key finding from on-chain data",
  "sentiment_analysis": "News/sentiment interpretation",
  "key_risks": ["risk1", "risk2", "risk3"],
  "signal": "LONG | SHORT | HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed reasoning chain",
  "entry_rationale": "Specific entry rationale if LONG/SHORT, else null",
  "invalidation_condition": "What would invalidate this signal"
}"""


@dataclass
class LLMAnalysisResult:
    market_summary: str
    btc_impact: str
    onchain_insight: str
    sentiment_analysis: str
    key_risks: list[str]
    signal: str  # "LONG" | "SHORT" | "HOLD"
    confidence: float
    reasoning: str
    entry_rationale: Optional[str]
    invalidation_condition: str
    provider_used: str  # "groq" | "gemini"

    @classmethod
    def empty(cls) -> "LLMAnalysisResult":
        return cls(
            market_summary="Analysis unavailable",
            btc_impact="N/A",
            onchain_insight="N/A",
            sentiment_analysis="N/A",
            key_risks=["System error — analysis incomplete"],
            signal="HOLD",
            confidence=0.0,
            reasoning="Pipeline failed before LLM analysis",
            entry_rationale=None,
            invalidation_condition="N/A",
            provider_used="none",
        )


class LLMEngine:
    """
    Dual-provider LLM engine.
    
    Strategy: Groq-first karena latency lebih rendah (inferensi di hardware mereka).
    Gemini sebagai fallback jika Groq rate-limit atau down.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Separate rate limiters per provider
        self._groq_limiter = RateLimiter(
            max_calls=settings.GROQ_RPM_LIMIT,
            period_seconds=60,
        )
        self._http_client = httpx.AsyncClient(timeout=30.0)

    async def analyze(self, context: FullAnalysisContext) -> LLMAnalysisResult:
        """
        Analisis dengan fallback chain: Groq → Gemini → Empty.
        """
        user_prompt = self._build_prompt(context)

        # Try Groq first
        try:
            async with self._groq_limiter:
                result = await self._call_groq(user_prompt)
                return result
        except Exception as e:
            logger.warning(f"Groq failed ({e}), falling back to Gemini...")

        # Fallback to Gemini
        try:
            result = await self._call_gemini(user_prompt)
            return result
        except Exception as e:
            logger.error(f"Gemini also failed ({e}). Returning empty result.")
            return LLMAnalysisResult.empty()

    def _build_prompt(self, ctx: FullAnalysisContext) -> str:
        """
        Membangun prompt yang kaya data.
        Struktur: BTC Anchor → On-Chain → News → Task.
        """
        prompt_parts = [f"# Trading Analysis Request: {ctx.symbol}"]
        prompt_parts.append(f"Timestamp: {ctx.timestamp.isoformat()}")

        # BTC Correlation Section
        if ctx.btc_state:
            prompt_parts.append(f"""
## Bitcoin Market Anchor (CRITICAL — Altcoin Context)
- BTC Price: ${ctx.btc_state.price:,.2f}
- 24h Change: {ctx.btc_state.change_24h_pct:+.2f}%
- BTC Trend: {ctx.btc_state.trend}
- BTC Dominance: {ctx.btc_state.dominance_pct:.1f}%
- BTC Volume 24h: ${ctx.btc_state.volume_24h_usdt:,.0f} USDT

⚠️ You MUST factor BTC trend into your {ctx.symbol} analysis.
""")
        else:
            prompt_parts.append("\n## Analyzing BTC directly — no correlation anchor needed.\n")

        # On-Chain Section
        prompt_parts.append(f"""
## On-Chain Data (Real-Time)
- Whale Transactions (last hour): {len(ctx.onchain.whale_transactions)}
- Top Whale Moves: {self._format_whale_txns(ctx.onchain.whale_transactions[:5])}
- DEX Liquidity (USD): ${ctx.onchain.dex_liquidity_usd:,.0f}
- Volume Anomaly Score: {ctx.onchain.volume_anomaly_score:.2f} (>1.5 = anomalous)
- Large Transfers: {len(ctx.onchain.large_transfers)} detected
""")

        # Sentiment Section
        prompt_parts.append(f"""
## Market Sentiment & News
- Computed Sentiment Score: {ctx.sentiment_score:.1f}/100 
  (interpretation: <-40=Extreme Fear, -40 to -10=Fear, -10 to 10=Neutral, 10 to 40=Greed, >40=Extreme Greed)
- Recent Headlines:
{self._format_news(ctx.news.items[:8])}
""")

        prompt_parts.append(f"""
## Your Task
Synthesize ALL data above and produce your structured JSON trading analysis for {ctx.symbol}.
Consider: data confluence (multiple signals agreeing), risk/reward, and BTC macro context.
""")

        return "\n".join(prompt_parts)

    async def _call_groq(self, user_prompt: str) -> LLMAnalysisResult:
        """Direct HTTP call ke Groq API (OpenAI-compatible endpoint)."""
        headers = {
            "Authorization": f"Bearer {self.settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.settings.LLM_MAX_TOKENS,
            "temperature": self.settings.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }

        response = await self._http_client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()

        raw_json = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(raw_json)
        return self._parse_llm_response(parsed, provider="groq")

    async def _call_gemini(self, user_prompt: str) -> LLMAnalysisResult:
        """Direct HTTP call ke Google Gemini API."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.GEMINI_MODEL}:generateContent"
            f"?key={self.settings.GEMINI_API_KEY}"
        )
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{ANALYSIS_SYSTEM_PROMPT}\n\n{user_prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": self.settings.LLM_TEMPERATURE,
                "maxOutputTokens": self.settings.LLM_MAX_TOKENS,
                "responseMimeType": "application/json",
            },
        }

        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()

        raw_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(raw_text)
        return self._parse_llm_response(parsed, provider="gemini")

    def _parse_llm_response(self, data: dict, provider: str) -> LLMAnalysisResult:
        """Validasi dan mapping response LLM ke dataclass."""
        return LLMAnalysisResult(
            market_summary=data.get("market_summary", ""),
            btc_impact=data.get("btc_impact", "N/A"),
            onchain_insight=data.get("onchain_insight", ""),
            sentiment_analysis=data.get("sentiment_analysis", ""),
            key_risks=data.get("key_risks", []),
            signal=data.get("signal", "HOLD").upper(),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            entry_rationale=data.get("entry_rationale"),
            invalidation_condition=data.get("invalidation_condition", ""),
            provider_used=provider,
        )

    @staticmethod
    def _format_whale_txns(txns: list) -> str:
        if not txns:
            return "None detected"
        return "; ".join([f"${t.amount_usd:,.0f} {t.direction}" for t in txns])

    @staticmethod
    def _format_news(items: list) -> str:
        if not items:
            return "No recent news"
        return "\n".join([f"  - [{item.source}] {item.title}" for item in items])