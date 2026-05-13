# core/decision_engine.py
"""
Decision Engine — Multi-factor signal aggregation.

Filosofi:
  Tidak ada single indicator yang cukup untuk keputusan trading.
  Engine ini mengaggregasi signal dari 4 dimensi:
    1. LLM Reasoning Score      (bobot: 40%)
    2. Sentiment Score           (bobot: 20%)
    3. On-Chain Signal           (bobot: 25%)
    4. BTC Correlation Signal    (bobot: 15%)  — hanya untuk altcoins

  Final action diambil hanya jika:
    - Weighted confidence >= CONFIDENCE_THRESHOLD
    - Tidak ada critical override (e.g., extreme fear + bearish BTC)

Output: TradingSignal dengan action, entry, TP, SL, dan reasoning chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from config import Settings
from core.llm_engine import LLMAnalysisResult
from models.schemas import FullAnalysisContext
from utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds
CONFIDENCE_THRESHOLD = 0.62      # Minimum confidence untuk entry
SENTIMENT_EXTREME_FEAR = -60.0   # Di bawah ini: hindari LONG
SENTIMENT_EXTREME_GREED = 75.0   # Di atas ini: hindari SHORT (contrarian)
VOLUME_ANOMALY_BULLISH = 2.0     # Score > 2x avg = potensi breakout
WHALE_ACCUMULATION_THRESHOLD = 3 # Jumlah whale buys untuk dianggap akumulasi


class SignalAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


@dataclass
class SignalFactor:
    """Satu dimensi analisis dengan score dan reasoning."""
    name: str
    score: float          # -1.0 to +1.0 (negatif = bearish, positif = bullish)
    weight: float         # Bobot dalam weighted average
    reasoning: str
    override: bool = False  # Jika True, factor ini bisa veto keputusan final


@dataclass
class TradingSignal:
    """Output final decision engine."""
    action: SignalAction
    confidence: float           # 0.0 - 1.0
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_reward_ratio: float
    reasoning: str
    factors: list[SignalFactor] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @classmethod
    def hold(cls, reason: str) -> "TradingSignal":
        """Factory untuk HOLD signal cepat."""
        return cls(
            action=SignalAction.HOLD,
            confidence=0.0,
            entry_price=0.0,
            take_profit=0.0,
            stop_loss=0.0,
            risk_reward_ratio=0.0,
            reasoning=reason,
        )


class DecisionEngine:
    """
    Multi-factor weighted signal aggregator.
    
    Design: Pure function logic (tidak ada I/O, tidak ada state).
    Mudah di-unit test karena semua input explicit.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_signal(
        self,
        context: FullAnalysisContext,
        llm_result: LLMAnalysisResult,
    ) -> TradingSignal:
        """
        Entry point utama. Agregasi semua faktor → final signal.
        """
        logger.info(f"⚡ Generating signal for {context.symbol}...")

        # ── Hitung setiap faktor ───────────────────────────────────────────
        factors: list[SignalFactor] = []

        # Factor 1: LLM Score (40% weight)
        llm_factor = self._compute_llm_factor(llm_result)
        factors.append(llm_factor)

        # Factor 2: Sentiment Score (20% weight)
        sentiment_factor = self._compute_sentiment_factor(context.sentiment_score)
        factors.append(sentiment_factor)

        # Factor 3: On-Chain Signal (25% weight)
        onchain_factor = self._compute_onchain_factor(context)
        factors.append(onchain_factor)

        # Factor 4: BTC Correlation (15% weight) — hanya untuk altcoins
        if context.btc_state is not None:
            btc_factor = self._compute_btc_correlation_factor(context)
            factors.append(btc_factor)

        # ── Weighted aggregation ───────────────────────────────────────────
        weighted_score = self._weighted_aggregate(factors)

        # ── Override checks ────────────────────────────────────────────────
        override_result = self._check_overrides(factors, context, llm_result)
        if override_result is not None:
            logger.info(f"🚫 Signal override: {override_result}")
            return TradingSignal.hold(reason=override_result)

        # ── Determine action ───────────────────────────────────────────────
        action = self._score_to_action(weighted_score)
        confidence = abs(weighted_score)

        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(
                f"⏸️  Confidence {confidence:.2f} < threshold {CONFIDENCE_THRESHOLD} → HOLD"
            )
            return TradingSignal.hold(
                reason=f"Insufficient confluence (confidence={confidence:.2f})"
            )

        # ── Calculate levels ───────────────────────────────────────────────
        # Ambil current price dari on-chain/market data
        # Untuk sekarang kita compute TP/SL dari LLM entry rationale
        entry_price = self._estimate_entry_price(context, llm_result)
        take_profit, stop_loss = self._calculate_levels(
            entry_price=entry_price,
            action=action,
            rr_ratio=self.settings.DEFAULT_RR_RATIO,
        )

        signal = TradingSignal(
            action=action,
            confidence=round(confidence, 4),
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            risk_reward_ratio=self.settings.DEFAULT_RR_RATIO,
            reasoning=self._build_reasoning(factors, weighted_score, action),
            factors=factors,
        )

        logger.info(
            f"✅ Signal: {action.value} | confidence={confidence:.2f} | "
            f"entry={entry_price:.4f} | TP={take_profit:.4f} | SL={stop_loss:.4f}"
        )
        return signal

    # ── Factor Computers ───────────────────────────────────────────────────────

    def _compute_llm_factor(self, llm_result: LLMAnalysisResult) -> SignalFactor:
        """
        Map LLM signal + confidence → normalized score.
        LONG @ 0.8 confidence → score = +0.8
        SHORT @ 0.7 confidence → score = -0.7
        HOLD → score = 0.0
        """
        llm_signal = llm_result.signal.upper()
        raw_confidence = llm_result.confidence

        if llm_signal == "LONG":
            score = raw_confidence
        elif llm_signal == "SHORT":
            score = -raw_confidence
        else:
            score = 0.0

        return SignalFactor(
            name="LLM_REASONING",
            score=score,
            weight=0.40,
            reasoning=(
                f"LLM ({llm_result.provider_used}) signals {llm_signal} "
                f"with {raw_confidence:.0%} confidence. "
                f"Key: {llm_result.reasoning[:200]}..."
            ),
        )

    def _compute_sentiment_factor(self, sentiment_score: float) -> SignalFactor:
        """
        Normalize sentiment score (-100 to +100) → factor score (-1 to +1).

        Penting: Sentiment adalah lagging indicator.
        Extreme greed bisa berarti reversal akan datang (contrarian).
        """
        # Normalize ke -1..+1
        normalized = sentiment_score / 100.0

        # Contrarian adjustment di extreme zones
        if sentiment_score > SENTIMENT_EXTREME_GREED:
            normalized *= 0.3  # Reduce bullish signal saat extreme greed
            reasoning = f"CONTRARIAN: Extreme greed ({sentiment_score:.0f}) → dampening bullish signal"
        elif sentiment_score < SENTIMENT_EXTREME_FEAR:
            normalized *= 0.3  # Reduce bearish signal saat extreme fear
            reasoning = f"CONTRARIAN: Extreme fear ({sentiment_score:.0f}) → dampening bearish signal"
        else:
            reasoning = f"Sentiment score: {sentiment_score:.0f}/100 → normalized: {normalized:+.2f}"

        return SignalFactor(
            name="SENTIMENT",
            score=normalized,
            weight=0.20,
            reasoning=reasoning,
        )

    def _compute_onchain_factor(self, context: FullAnalysisContext) -> SignalFactor:
        """
        Hitung signal dari on-chain data.
        
        Logic:
          - Whale net buys ↑ → bullish
          - Volume anomaly tinggi + liquidity tinggi → breakout potential
          - Large outflows (exchange deposits) → bearish
        """
        onchain = context.onchain
        score = 0.0
        signals = []

        # Whale sentiment
        whale_buys = sum(
            1 for t in onchain.whale_transactions if t.direction == "IN"
        )
        whale_sells = sum(
            1 for t in onchain.whale_transactions if t.direction == "OUT"
        )
        total_whales = whale_buys + whale_sells

        if total_whales > 0:
            whale_ratio = (whale_buys - whale_sells) / total_whales
            score += whale_ratio * 0.5  # Max 0.5 contribution dari whale signal
            signals.append(
                f"Whale net: +{whale_buys}buy/-{whale_sells}sell (ratio={whale_ratio:+.2f})"
            )

        # Volume anomaly
        if onchain.volume_anomaly_score > VOLUME_ANOMALY_BULLISH:
            # Volume spike: bullish jika whale buys juga tinggi, otherwise neutral
            multiplier = 0.3 if whale_buys >= whale_sells else 0.0
            score += multiplier
            signals.append(
                f"Volume spike {onchain.volume_anomaly_score:.1f}x normal"
            )

        # Liquidity assessment
        if onchain.dex_liquidity_usd > 5_000_000:  # > $5M = healthy
            score += 0.1
            signals.append(f"Healthy DEX liquidity: ${onchain.dex_liquidity_usd:,.0f}")
        elif onchain.dex_liquidity_usd < 500_000:  # < $500K = risky
            score -= 0.2
            signals.append(f"⚠️ Low liquidity: ${onchain.dex_liquidity_usd:,.0f}")

        # Clamp ke -1..+1
        score = max(-1.0, min(1.0, score))

        return SignalFactor(
            name="ON_CHAIN",
            score=score,
            weight=0.25,
            reasoning=" | ".join(signals) if signals else "Insufficient on-chain data",
        )

    def _compute_btc_correlation_factor(self, context: FullAnalysisContext) -> SignalFactor:
        """
        Pilar 3: BTC sebagai market anchor untuk altcoins.

        Rules:
          - BTC BULLISH  → altcoin bullish bias +0.5
          - BTC BEARISH  → altcoin bearish bias -0.5 (hard override candidate)
          - BTC SIDEWAYS → neutral, 0.0
          - BTC drop >5% in 24h → bearish override flag
        """
        btc = context.btc_state
        score = 0.0
        override = False

        if btc.trend == "BULLISH":
            score = 0.5
            reasoning = f"BTC trending BULLISH (+{btc.change_24h_pct:.2f}% 24h) → altcoin tailwind"
        elif btc.trend == "BEARISH":
            score = -0.5
            reasoning = f"BTC trending BEARISH ({btc.change_24h_pct:.2f}% 24h) → altcoin headwind"
        else:
            score = 0.0
            reasoning = "BTC SIDEWAYS → neutral correlation impact"

        # Emergency override: BTC crash > 5%
        if btc.change_24h_pct < -5.0:
            score = -1.0
            override = True
            reasoning = (
                f"🚨 BTC CRASH DETECTED: {btc.change_24h_pct:.2f}% drop in 24h. "
                f"Override: force HOLD on all altcoins."
            )

        return SignalFactor(
            name="BTC_CORRELATION",
            score=score,
            weight=0.15,
            reasoning=reasoning,
            override=override,
        )

    # ── Aggregation & Override Logic ───────────────────────────────────────────

    @staticmethod
    def _weighted_aggregate(factors: list[SignalFactor]) -> float:
        """
        Weighted average semua factors.
        Total weight harus = 1.0 (dinormalisasi jika tidak).
        """
        total_weight = sum(f.weight for f in factors)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(f.score * f.weight for f in factors)
        return weighted_sum / total_weight  # Normalize jika weights != 1.0

    @staticmethod
    def _check_overrides(
        factors: list[SignalFactor],
        context: FullAnalysisContext,
        llm_result: LLMAnalysisResult,
    ) -> Optional[str]:
        """
        Cek kondisi yang memaksa HOLD regardless of score.
        Returns override reason string, atau None jika tidak ada override.
        """
        for factor in factors:
            if factor.override:
                return f"OVERRIDE by {factor.name}: {factor.reasoning}"

        # LLM dan on-chain bertentangan keras → uncertainty tinggi
        llm_bullish = llm_result.signal.upper() == "LONG"
        onchain_factor = next((f for f in factors if f.name == "ON_CHAIN"), None)

        if onchain_factor and llm_bullish and onchain_factor.score < -0.5:
            return (
                "CONFLICT: LLM bullish but on-chain strongly bearish. "
                "Insufficient confluence for entry."
            )

        if onchain_factor and not llm_bullish and onchain_factor.score > 0.5:
            return (
                "CONFLICT: LLM bearish but on-chain strongly bullish. "
                "Insufficient confluence for entry."
            )

        return None

    @staticmethod
    def _score_to_action(score: float) -> SignalAction:
        """Map weighted score → action."""
        if score > 0:
            return SignalAction.LONG
        elif score < 0:
            return SignalAction.SHORT
        return SignalAction.HOLD

    @staticmethod
    def _estimate_entry_price(
        context: FullAnalysisContext,
        llm_result: LLMAnalysisResult,
    ) -> float:
        """
        Entry price estimation.
        
        Dalam real implementation: ambil dari live market ticker.
        Untuk sekarang: kita gunakan placeholder dari context.
        Implementasi penuh ada di data/market.py -> get_ticker().
        """
        # TODO: inject market_client dan ambil current price
        # Sementara return 0.0 — akan diisi saat market.py diintegrasikan
        return 0.0

    @staticmethod
    def _calculate_levels(
        entry_price: float,
        action: SignalAction,
        rr_ratio: float,
        atr_pct: float = 0.02,  # 2% ATR estimasi default
    ) -> tuple[float, float]:
        """
        Hitung TP dan SL berdasarkan entry price dan RR ratio.

        Formula:
          SL distance = entry_price * atr_pct
          TP distance = SL distance * rr_ratio

          LONG : TP = entry + TP_dist | SL = entry - SL_dist
          SHORT: TP = entry - TP_dist | SL = entry + SL_dist
        """
        if entry_price == 0:
            return 0.0, 0.0

        sl_distance = entry_price * atr_pct
        tp_distance = sl_distance * rr_ratio

        if action == SignalAction.LONG:
            take_profit = entry_price + tp_distance
            stop_loss = entry_price - sl_distance
        else:  # SHORT
            take_profit = entry_price - tp_distance
            stop_loss = entry_price + sl_distance

        return round(take_profit, 6), round(stop_loss, 6)

    @staticmethod
    def _build_reasoning(
        factors: list[SignalFactor],
        weighted_score: float,
        action: SignalAction,
    ) -> str:
        """Build human-readable reasoning chain."""
        lines = [
            f"Action: {action.value} | Weighted Score: {weighted_score:+.4f}",
            "─" * 50,
        ]
        for f in factors:
            direction = "↑ BULL" if f.score > 0 else ("↓ BEAR" if f.score < 0 else "→ NEUT")
            lines.append(
                f"[{f.name}] w={f.weight:.0%} | score={f.score:+.2f} | {direction}"
            )
            lines.append(f"  └ {f.reasoning}")

        return "\n".join(lines)