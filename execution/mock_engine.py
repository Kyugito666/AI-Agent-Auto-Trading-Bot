# execution/mock_engine.py
"""
Mock Execution Engine — Paper Trading (Dry Run Mode).

HARD CONSTRAINT: Modul ini DILARANG KERAS memiliki import apapun dari
library exchange (ccxt, python-binance, pybit). Tidak ada koneksi ke
exchange nyata dalam kondisi apapun.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from config import Settings
from core.decision_engine import TradingSignal
from execution.risk_manager import RiskManager
from models.database import AsyncSessionLocal
from models.orm_models import PaperTrade, TradeStatus
from models.schemas import FullAnalysisContext
from utils.logger import get_logger

logger = get_logger(__name__)


class MockExecutionEngine:
    """
    Mensimulasikan eksekusi order tanpa koneksi ke exchange manapun.
    
    Semua "trade" disimpan ke SQLite sebagai record simulasi.
    PnL dihitung berdasarkan harga aktual yang di-fetch berikutnya.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.risk_manager = RiskManager(settings=settings)
        self._current_capital = settings.PAPER_CAPITAL

    async def initialize(self) -> None:
        """Load capital state dari DB jika ada run sebelumnya."""
        async with AsyncSessionLocal() as session:
            last_capital = await self._get_last_capital(session)
            if last_capital:
                self._current_capital = last_capital
                logger.info(f"💰 Paper capital restored: ${self._current_capital:,.2f}")
            else:
                logger.info(f"💰 Starting fresh paper capital: ${self._current_capital:,.2f}")

    async def execute_paper_trade(
        self,
        symbol: str,
        signal: TradingSignal,
        context: FullAnalysisContext,
    ) -> str:
        """
        Buat dan simpan simulasi trade ke database.
        
        Returns:
            trade_id: UUID string dari trade yang dibuat
        """
        if signal.action not in ("LONG", "SHORT"):
            raise ValueError(f"execute_paper_trade dipanggil dengan signal {signal.action}")

        # Hitung position sizing berdasarkan risk management
        position = self.risk_manager.calculate_position(
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            capital=self._current_capital,
            risk_pct=self.settings.DEFAULT_RISK_PCT,
        )

        trade_id = str(uuid.uuid4())

        trade = PaperTrade(
            id=trade_id,
            symbol=symbol,
            action=signal.action,
            entry_price=signal.entry_price,
            take_profit=signal.take_profit,
            stop_loss=signal.stop_loss,
            quantity=position.quantity,
            notional_value=position.notional_value,
            risk_amount=position.risk_amount,
            leverage=self.settings.DEFAULT_LEVERAGE,
            status=TradeStatus.OPEN,
            confidence=signal.confidence,
            signal_reasoning=signal.reasoning,
            llm_provider=context.llm_analysis.provider_used if hasattr(context, 'llm_analysis') else "unknown",
            sentiment_score=context.sentiment_score,
            btc_trend=context.btc_state.trend if context.btc_state else "N/A",
            opened_at=datetime.now(tz=timezone.utc),
            capital_before=self._current_capital,
        )

        async with AsyncSessionLocal() as session:
            async with session.begin():
                session.add(trade)

        logger.info(
            f"📝 PAPER TRADE OPENED\n"
            f"   ID: {trade_id}\n"
            f"   {symbol} {signal.action} @ ${signal.entry_price:,.4f}\n"
            f"   Qty: {position.quantity:.6f} | Notional: ${position.notional_value:,.2f}\n"
            f"   TP: ${signal.take_profit:,.4f} | SL: ${signal.stop_loss:,.4f}\n"
            f"   Risk: ${position.risk_amount:,.2f} ({self.settings.DEFAULT_RISK_PCT*100:.1f}%)\n"
            f"   Capital: ${self._current_capital:,.2f}"
        )

        return trade_id

    async def update_trade_pnl(self, trade_id: str, current_price: float) -> Optional[float]:
        """
        Update PnL paper trade berdasarkan harga terkini.
        Otomatis close jika TP atau SL tersentuh.
        
        Dipanggil oleh scheduler secara periodik.
        """
        async with AsyncSessionLocal() as session:
            async with session.begin():
                trade = await session.get(PaperTrade, trade_id)
                if not trade or trade.status != TradeStatus.OPEN:
                    return None

                # Hitung unrealized PnL
                if trade.action == "LONG":
                    pnl = (current_price - trade.entry_price) * trade.quantity
                    tp_hit = current_price >= trade.take_profit
                    sl_hit = current_price <= trade.stop_loss
                else:  # SHORT
                    pnl = (trade.entry_price - current_price) * trade.quantity
                    tp_hit = current_price <= trade.take_profit
                    sl_hit = current_price >= trade.stop_loss

                trade.current_price = current_price
                trade.unrealized_pnl = pnl

                if tp_hit or sl_hit:
                    trade.status = TradeStatus.CLOSED
                    trade.close_reason = "TP_HIT" if tp_hit else "SL_HIT"
                    trade.realized_pnl = pnl
                    trade.closed_at = datetime.now(tz=timezone.utc)
                    self._current_capital += pnl

                    logger.info(
                        f"🏁 PAPER TRADE CLOSED | {trade.symbol} | "
                        f"Reason: {trade.close_reason} | "
                        f"PnL: ${pnl:+,.2f} | "
                        f"New Capital: ${self._current_capital:,.2f}"
                    )

                return pnl

    @staticmethod
    async def _get_last_capital(session: AsyncSession) -> Optional[float]:
        """Ambil capital terakhir dari trade record terbaru."""
        from sqlalchemy import select, desc
        result = await session.execute(
            select(PaperTrade.capital_before)
            .order_by(desc(PaperTrade.opened_at))
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row