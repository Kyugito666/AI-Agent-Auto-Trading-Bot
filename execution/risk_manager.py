# execution/risk_manager.py
"""
Risk Manager — Position sizing dan risk calculation.

Mengimplementasikan fixed fractional position sizing:
  Risk per trade = Capital × Risk%
  Position size  = Risk Amount / (Entry - Stop Loss)

Ini adalah standar industri untuk risk management sistematis.
Tidak ada "feeling" — semua matematis.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionResult:
    """Output kalkulasi position sizing."""
    quantity: float          # Berapa unit yang dibeli/dijual
    notional_value: float    # Total nilai posisi (quantity × entry_price)
    risk_amount: float       # Modal yang di-risk (dalam USDT)
    stop_loss_pct: float     # Jarak SL dari entry dalam persen
    take_profit_pct: float   # Jarak TP dari entry dalam persen
    risk_reward_ratio: float
    max_loss_scenario: float  # Worst case jika SL hit
    max_gain_scenario: float  # Best case jika TP hit
    capital_utilization_pct: float  # % dari capital yang digunakan


@dataclass
class RiskValidation:
    """Hasil validasi risk sebelum eksekusi."""
    is_valid: bool
    warnings: list[str]
    errors: list[str]


class RiskManager:
    """
    Fixed Fractional Position Sizing.

    Invariant: Risk per trade TIDAK PERNAH melebihi MAX_RISK_PCT dari capital.
    """

    # Hard limits — tidak bisa di-override dari settings
    MAX_RISK_PCT = 0.05          # Maximum 5% risk per trade (hard cap)
    MAX_CAPITAL_UTIL_PCT = 0.30  # Maximum 30% capital dalam satu posisi
    MIN_NOTIONAL_USDT = 10.0     # Minimum order size

    def __init__(self, settings: Settings):
        self.settings = settings

    def calculate_position(
        self,
        entry_price: float,
        stop_loss: float,
        capital: float,
        risk_pct: float | None = None,
        take_profit: float | None = None,
    ) -> PositionResult:
        """
        Hitung ukuran posisi berdasarkan risk tolerance.

        Args:
            entry_price : Harga masuk rencana
            stop_loss   : Level stop loss
            capital     : Modal tersedia (USDT)
            risk_pct    : Persentase capital yang di-risk (default: dari settings)
            take_profit : Level take profit (optional, untuk kalkulasi RR)

        Returns:
            PositionResult dengan semua kalkulasi risk lengkap
        """
        # Validasi dasar
        if entry_price <= 0 or stop_loss <= 0 or capital <= 0:
            logger.warning("Invalid inputs untuk position calculation")
            return self._zero_position()

        # Gunakan settings risk_pct jika tidak diberikan
        effective_risk_pct = min(
            risk_pct or self.settings.DEFAULT_RISK_PCT,
            self.MAX_RISK_PCT  # Hard cap
        )

        # ── Core calculation ───────────────────────────────────────────────
        risk_amount = capital * effective_risk_pct

        # Jarak SL dari entry (dalam unit harga)
        sl_distance = abs(entry_price - stop_loss)
        if sl_distance == 0:
            logger.warning("SL distance = 0, cannot calculate position size")
            return self._zero_position()

        # Position quantity = risk_amount / sl_distance
        # Contoh: risk $200 / SL distance $0.50 = 400 unit
        quantity = risk_amount / sl_distance

        # Notional value (total exposure)
        notional_value = quantity * entry_price

        # ── Capital utilization guard ──────────────────────────────────────
        max_notional = capital * self.MAX_CAPITAL_UTIL_PCT
        if notional_value > max_notional:
            # Scale down agar tidak exceed max utilization
            scale_factor = max_notional / notional_value
            quantity *= scale_factor
            notional_value = max_notional
            risk_amount = quantity * sl_distance
            logger.debug(
                f"Position scaled down by {scale_factor:.2f}x "
                f"to respect max capital utilization"
            )

        # ── Percentage calculations ────────────────────────────────────────
        sl_pct = (sl_distance / entry_price) * 100

        # TP calculations
        tp_pct = 0.0
        rr_ratio = 0.0
        max_gain = 0.0

        if take_profit:
            tp_distance = abs(take_profit - entry_price)
            tp_pct = (tp_distance / entry_price) * 100
            rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0.0
            max_gain = quantity * tp_distance
        else:
            # Hitung dari RR ratio default
            rr_ratio = self.settings.DEFAULT_RR_RATIO
            tp_pct = sl_pct * rr_ratio
            max_gain = risk_amount * rr_ratio

        capital_util_pct = (notional_value / capital) * 100

        result = PositionResult(
            quantity=round(quantity, 8),
            notional_value=round(notional_value, 2),
            risk_amount=round(risk_amount, 2),
            stop_loss_pct=round(sl_pct, 4),
            take_profit_pct=round(tp_pct, 4),
            risk_reward_ratio=round(rr_ratio, 2),
            max_loss_scenario=round(risk_amount, 2),
            max_gain_scenario=round(max_gain, 2),
            capital_utilization_pct=round(capital_util_pct, 2),
        )

        logger.debug(
            f"Position calculated | qty={result.quantity:.6f} | "
            f"notional=${result.notional_value:,.2f} | "
            f"risk=${result.risk_amount:,.2f} ({effective_risk_pct*100:.1f}%) | "
            f"RR={result.risk_reward_ratio:.1f}:1 | "
            f"capital_util={result.capital_utilization_pct:.1f}%"
        )
        return result

    def validate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        capital: float,
        position: PositionResult,
    ) -> RiskValidation:
        """
        Validasi semua parameter trade sebelum eksekusi paper order.
        Returns RiskValidation dengan list errors dan warnings.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Hard errors (trade HARUS ditolak) ─────────────────────────────
        if position.notional_value < self.MIN_NOTIONAL_USDT:
            errors.append(
                f"Notional value ${position.notional_value:.2f} "
                f"< minimum ${self.MIN_NOTIONAL_USDT}"
            )

        if position.risk_amount > capital * self.MAX_RISK_PCT:
            errors.append(
                f"Risk amount ${position.risk_amount:.2f} "
                f"exceeds maximum {self.MAX_RISK_PCT*100:.0f}% of capital"
            )

        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            errors.append("Entry, SL, atau TP price tidak valid (≤ 0)")

        if position.risk_reward_ratio < 1.0:
            errors.append(
                f"RR ratio {position.risk_reward_ratio:.2f} < 1.0. "
                f"Trade tidak worthwhile secara matematis."
            )

        # ── Warnings (trade bisa jalan, tapi perlu perhatian) ─────────────
        if position.capital_utilization_pct > 20.0:
            warnings.append(
                f"Capital utilization tinggi: {position.capital_utilization_pct:.1f}% "
                f"(recommended: < 20%)"
            )

        if position.stop_loss_pct > 5.0:
            warnings.append(
                f"SL distance {position.stop_loss_pct:.2f}% relatif lebar. "
                f"Pertimbangkan SL lebih ketat."
            )

        if position.risk_reward_ratio < self.settings.DEFAULT_RR_RATIO:
            warnings.append(
                f"RR {position.risk_reward_ratio:.1f}:1 di bawah target "
                f"{self.settings.DEFAULT_RR_RATIO}:1"
            )

        if capital < 100:
            warnings.append("Capital rendah (< $100). Sizing sangat terbatas.")

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(f"Trade validation FAILED: {errors}")
        elif warnings:
            logger.info(f"Trade validation PASSED with warnings: {warnings}")
        else:
            logger.info("Trade validation PASSED cleanly")

        return RiskValidation(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
        )

    def calculate_portfolio_metrics(
        self,
        open_trades: list[dict],
        capital: float,
    ) -> dict:
        """
        Hitung metrics portfolio-level dari semua open paper trades.
        Berguna untuk monitoring dashboard.
        """
        if not open_trades:
            return {
                "total_exposure": 0.0,
                "total_risk_amount": 0.0,
                "exposure_pct": 0.0,
                "trade_count": 0,
                "avg_rr": 0.0,
            }

        total_exposure = sum(t.get("notional_value", 0) for t in open_trades)
        total_risk = sum(t.get("risk_amount", 0) for t in open_trades)
        avg_rr = (
            sum(t.get("risk_reward_ratio", 0) for t in open_trades) / len(open_trades)
        )

        return {
            "total_exposure": round(total_exposure, 2),
            "total_risk_amount": round(total_risk, 2),
            "exposure_pct": round((total_exposure / capital) * 100, 2),
            "risk_pct": round((total_risk / capital) * 100, 2),
            "trade_count": len(open_trades),
            "avg_rr": round(avg_rr, 2),
        }

    @staticmethod
    def _zero_position() -> PositionResult:
        return PositionResult(
            quantity=0.0,
            notional_value=0.0,
            risk_amount=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            risk_reward_ratio=0.0,
            max_loss_scenario=0.0,
            max_gain_scenario=0.0,
            capital_utilization_pct=0.0,
        )