# core/analyzer.py
"""
TradingAnalyzer — Orchestrator utama seluruh pipeline analisis.

Pipeline Flow:
  ┌─────────────────────────────────────────────────────────┐
  │  run_full_pipeline(symbol)                              │
  │                                                         │
  │  1. [BTC Correlation Check] ─── Is symbol BTC? ──No──► │
  │                                    │                    │
  │                                   Yes                   │
  │                                    ▼                    │
  │  2. [BTC Anchor Analysis] ← Fetch BTC market state      │
  │                                    │                    │
  │                                    ▼                    │
  │  3. [On-Chain Data Fetch]  ← Whale moves, liquidity     │
  │                                    │                    │
  │                                    ▼                    │
  │  4. [News & Sentiment]     ← CryptoPanic + scraper      │
  │                                    │                    │
  │                                    ▼                    │
  │  5. [LLM Deep Think]       ← Semua data → LLM prompt   │
  │                                    │                    │
  │                                    ▼                    │
  │  6. [Decision Engine]      ← Signal aggregation         │
  │                                    │                    │
  │                                    ▼                    │
  │  7. [Mock Execution]       ← Log paper trade ke DB      │
  └─────────────────────────────────────────────────────────┘
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import Settings
from core.decision_engine import DecisionEngine, TradingSignal
from core.llm_engine import LLMEngine, LLMAnalysisResult
from core.sentiment import SentimentScorer
from data.market import MarketDataClient
from data.onchain import OnChainDataClient
from data.research import ResearchClient
from execution.mock_engine import MockExecutionEngine
from models.schemas import (
    BTCMarketState,
    OnChainSnapshot,
    NewsSnapshot,
    FullAnalysisContext,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisPipelineResult:
    """Hasil lengkap satu siklus analisis untuk satu symbol."""
    symbol: str
    timestamp: datetime
    btc_state: Optional[BTCMarketState]
    onchain: OnChainSnapshot
    news: NewsSnapshot
    sentiment_score: float  # -100 to 100
    llm_analysis: LLMAnalysisResult
    signal: TradingSignal
    trade_id: Optional[str] = None  # Set jika mock order dieksekusi
    error: Optional[str] = None


class TradingAnalyzer:
    """
    Orchestrator utama. Mengkoordinasikan semua sub-modul.
    
    Design Decision: Dependency Injection pattern dipakai agar setiap
    sub-modul bisa di-mock secara independen dalam unit tests.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = False

        # Sub-modules (di-inject saat initialize())
        self.market_client: MarketDataClient
        self.onchain_client: OnChainDataClient
        self.research_client: ResearchClient
        self.sentiment_scorer: SentimentScorer
        self.llm_engine: LLMEngine
        self.decision_engine: DecisionEngine
        self.mock_executor: MockExecutionEngine

    async def initialize(self) -> None:
        """
        Async initialization. Constructor tidak bisa async,
        jadi pola 'two-phase init' digunakan.
        """
        if self._initialized:
            return

        self.market_client = MarketDataClient(settings=self.settings)
        self.onchain_client = OnChainDataClient(settings=self.settings)
        self.research_client = ResearchClient(settings=self.settings)
        self.sentiment_scorer = SentimentScorer()
        self.llm_engine = LLMEngine(settings=self.settings)
        self.decision_engine = DecisionEngine(settings=self.settings)
        self.mock_executor = MockExecutionEngine(settings=self.settings)

        await self.mock_executor.initialize()
        self._initialized = True
        logger.info("TradingAnalyzer initialized with all sub-modules")

    async def run_full_pipeline(self, symbol: str) -> AnalysisPipelineResult:
        """
        Eksekusi pipeline analisis lengkap untuk satu symbol.
        
        Raises: Tidak. Semua error dicatch dan dikembalikan dalam result.error
                agar scheduler tidak crash.
        """
        timestamp = datetime.now(tz=timezone.utc)
        logger.info(f"📊 Pipeline START | symbol={symbol}")

        try:
            # ── STEP 1 & 2: BTC Correlation Logic (Market Anchor) ─────────
            btc_state = await self._fetch_btc_anchor(symbol=symbol)

            # ── STEP 3: On-Chain Data ─────────────────────────────────────
            onchain_data = await self._fetch_onchain_data(symbol=symbol)

            # ── STEP 4: News & Sentiment ──────────────────────────────────
            news_data = await self._fetch_news_and_sentiment(symbol=symbol)

            # Sentiment scoring dari raw news
            sentiment_score = await self.sentiment_scorer.compute_score(
                news_items=news_data.items
            )

            # ── STEP 5: LLM Deep Think ────────────────────────────────────
            analysis_context = FullAnalysisContext(
                symbol=symbol,
                timestamp=timestamp,
                btc_state=btc_state,
                onchain=onchain_data,
                news=news_data,
                sentiment_score=sentiment_score,
            )
            llm_result = await self.llm_engine.analyze(context=analysis_context)

            # ── STEP 6: Decision Engine ───────────────────────────────────
            signal = await self.decision_engine.generate_signal(
                context=analysis_context,
                llm_result=llm_result,
            )

            # ── STEP 7: Mock Execution ────────────────────────────────────
            trade_id = None
            if signal.action in ("LONG", "SHORT"):
                trade_id = await self.mock_executor.execute_paper_trade(
                    symbol=symbol,
                    signal=signal,
                    context=analysis_context,
                )
                logger.info(f"📝 Paper trade logged | id={trade_id} | action={signal.action}")
            else:
                logger.info(f"⏸️  Signal HOLD — no trade executed for {symbol}")

            result = AnalysisPipelineResult(
                symbol=symbol,
                timestamp=timestamp,
                btc_state=btc_state,
                onchain=onchain_data,
                news=news_data,
                sentiment_score=sentiment_score,
                llm_analysis=llm_result,
                signal=signal,
                trade_id=trade_id,
            )

            logger.info(
                f"✅ Pipeline COMPLETE | symbol={symbol} | action={signal.action} "
                f"| confidence={signal.confidence:.2f} | sentiment={sentiment_score:.1f}"
            )
            return result

        except Exception as e:
            logger.exception(f"❌ Pipeline FAILED | symbol={symbol} | error={e}")
            # Return partial result dengan error flag — jangan raise
            return AnalysisPipelineResult(
                symbol=symbol,
                timestamp=timestamp,
                btc_state=None,
                onchain=OnChainSnapshot.empty(),
                news=NewsSnapshot.empty(),
                sentiment_score=0.0,
                llm_analysis=LLMAnalysisResult.empty(),
                signal=TradingSignal.hold(reason=f"Pipeline error: {e}"),
                error=str(e),
            )

    # ── Private Pipeline Steps ─────────────────────────────────────────────────

    async def _fetch_btc_anchor(self, symbol: str) -> Optional[BTCMarketState]:
        """
        Pilar 3: Bitcoin Correlation Logic.
        
        Jika symbol BUKAN BTCUSDT → WAJIB fetch BTC state terlebih dahulu.
        Jika symbol BTCUSDT → skip (BTC is its own anchor).
        
        Design Rationale: Altcoins secara historis berkorelasi tinggi (0.7-0.9)
        dengan BTC. Mengabaikan tren BTC saat trading altcoin adalah risiko fatal.
        """
        if symbol == "BTCUSDT":
            logger.debug("Symbol is BTC — skipping BTC anchor fetch")
            return None  # BTC is its own anchor

        logger.info(f"⚓ Fetching BTC anchor for altcoin {symbol}...")

        # Fetch concurrently: ticker + klines untuk BTC
        btc_ticker, btc_klines = await asyncio.gather(
            self.market_client.get_ticker("BTCUSDT"),
            self.market_client.get_klines("BTCUSDT", interval="1h", limit=24),
        )

        # Hitung metrics BTC
        btc_prices = [k.close for k in btc_klines]
        btc_24h_change = (
            (btc_prices[-1] - btc_prices[0]) / btc_prices[0] * 100
            if btc_prices else 0.0
        )

        # Tentukan trend BTC menggunakan EMA crossover sederhana
        btc_trend = _compute_trend(prices=btc_prices)

        state = BTCMarketState(
            price=btc_ticker.last_price,
            change_24h_pct=btc_24h_change,
            trend=btc_trend,  # "BULLISH" | "BEARISH" | "SIDEWAYS"
            volume_24h_usdt=btc_ticker.volume_usdt,
            dominance_pct=await self.onchain_client.get_btc_dominance(),
            timestamp=datetime.now(tz=timezone.utc),
        )

        logger.info(
            f"⚓ BTC Anchor | price=${state.price:,.0f} | "
            f"24h={state.change_24h_pct:+.2f}% | trend={state.trend}"
        )
        return state

    async def _fetch_onchain_data(self, symbol: str) -> OnChainSnapshot:
        """
        Pilar 1: On-Chain Real-Time Data.
        Fetch secara parallel untuk efisiensi maksimal.
        """
        # Extract base token dari pair (e.g., "ETHUSDT" → "ETH")
        base_token = symbol.replace("USDT", "").replace("BUSD", "")

        logger.info(f"🔗 Fetching on-chain data for {base_token}...")

        # Semua fetch dijalankan concurrent — total waktu = slowest call, bukan sum
        results = await asyncio.gather(
            self.onchain_client.get_whale_transactions(token=base_token),
            self.onchain_client.get_dex_liquidity(token=base_token),
            self.onchain_client.get_volume_anomaly(token=base_token),
            self.onchain_client.get_large_transfers(token=base_token),
            return_exceptions=True,  # Jangan crash jika salah satu gagal
        )

        whale_txns, liquidity, volume_anomaly, large_transfers = results

        # Handle partial failures gracefully
        snapshot = OnChainSnapshot(
            token=base_token,
            whale_transactions=whale_txns if not isinstance(whale_txns, Exception) else [],
            dex_liquidity_usd=liquidity if not isinstance(liquidity, Exception) else 0.0,
            volume_anomaly_score=volume_anomaly if not isinstance(volume_anomaly, Exception) else 0.0,
            large_transfers=large_transfers if not isinstance(large_transfers, Exception) else [],
            timestamp=datetime.now(tz=timezone.utc),
        )

        logger.info(
            f"🔗 On-Chain | whale_txns={len(snapshot.whale_transactions)} "
            f"| liquidity=${snapshot.dex_liquidity_usd:,.0f} "
            f"| anomaly_score={snapshot.volume_anomaly_score:.2f}"
        )
        return snapshot

    async def _fetch_news_and_sentiment(self, symbol: str) -> NewsSnapshot:
        """Pilar 2: Live Research Pipeline."""
        base_token = symbol.replace("USDT", "")
        logger.info(f"📰 Fetching news for {base_token}...")

        # Fetch dari multiple sources secara parallel
        cryptopanic_news, scraped_news = await asyncio.gather(
            self.research_client.fetch_cryptopanic(token=base_token),
            self.research_client.fetch_scraped_news(token=base_token),
            return_exceptions=True,
        )

        all_items = []
        if not isinstance(cryptopanic_news, Exception):
            all_items.extend(cryptopanic_news)
        if not isinstance(scraped_news, Exception):
            all_items.extend(scraped_news)

        snapshot = NewsSnapshot(
            token=base_token,
            items=all_items,
            timestamp=datetime.now(tz=timezone.utc),
        )

        logger.info(f"📰 News fetched | total_items={len(snapshot.items)}")
        return snapshot


# ── Helper Functions ───────────────────────────────────────────────────────────

def _compute_trend(prices: list[float], short_period: int = 7, long_period: int = 21) -> str:
    """
    Simple EMA crossover untuk menentukan tren.
    Menggunakan EMA bukan SMA karena lebih responsif terhadap pergerakan terbaru.
    """
    if len(prices) < long_period:
        return "SIDEWAYS"

    def ema(data: list[float], period: int) -> float:
        k = 2 / (period + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = price * k + ema_val * (1 - k)
        return ema_val

    short_ema = ema(prices[-short_period:], short_period)
    long_ema = ema(prices[-long_period:], long_period)
    diff_pct = (short_ema - long_ema) / long_ema * 100

    if diff_pct > 0.5:
        return "BULLISH"
    elif diff_pct < -0.5:
        return "BEARISH"
    else:
        return "SIDEWAYS"