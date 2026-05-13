# repositories/trade_repository.py
"""
Trade Repository — Single source of truth untuk semua DB operations.

Pattern: Repository memisahkan business logic dari persistence layer.
Semua query SQLAlchemy ada di sini, bukan tersebar di core modules.

Method naming convention:
  get_*    → SELECT single
  list_*   → SELECT multiple
  create_* → INSERT
  update_* → UPDATE
  delete_* → DELETE
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import select, func, desc, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from models.orm_models import PaperTrade, TradeStatus
from utils.logger import get_logger

logger = get_logger(__name__)


class TradeRepository:
    """Repository untuk PaperTrade CRUD dan analytics queries."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # ── CREATE ─────────────────────────────────────────────────────────────────

    async def create(self, trade: PaperTrade) -> PaperTrade:
        self._session.add(trade)
        await self._session.flush()  # Get ID tanpa commit
        await self._session.refresh(trade)
        return trade

    # ── READ ───────────────────────────────────────────────────────────────────

    async def get_by_id(self, trade_id: str) -> Optional[PaperTrade]:
        return await self._session.get(PaperTrade, trade_id)

    async def list_open_trades(self, symbol: Optional[str] = None) -> list[PaperTrade]:
        """Semua trade yang masih OPEN, optionally filtered by symbol."""
        stmt = (
            select(PaperTrade)
            .where(PaperTrade.status == TradeStatus.OPEN)
            .order_by(desc(PaperTrade.opened_at))
        )
        if symbol:
            stmt = stmt.where(PaperTrade.symbol == symbol.upper())

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_closed_trades(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PaperTrade]:
        stmt = (
            select(PaperTrade)
            .where(PaperTrade.status == TradeStatus.CLOSED)
            .order_by(desc(PaperTrade.closed_at))
            .limit(limit)
            .offset(offset)
        )
        if symbol:
            stmt = stmt.where(PaperTrade.symbol == symbol.upper())

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_recent(self, hours: int = 24, limit: int = 100) -> list[PaperTrade]:
        """Trades dari N jam terakhir."""
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        stmt = (
            select(PaperTrade)
            .where(PaperTrade.opened_at >= since)
            .order_by(desc(PaperTrade.opened_at))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # ── UPDATE ─────────────────────────────────────────────────────────────────

    async def update_price_and_pnl(
        self,
        trade_id: str,
        current_price: float,
        unrealized_pnl: float,
    ) -> Optional[PaperTrade]:
        stmt = (
            update(PaperTrade)
            .where(PaperTrade.id == trade_id)
            .values(
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                updated_at=datetime.now(tz=timezone.utc),
            )
            .returning(PaperTrade)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def close_trade(
        self,
        trade_id: str,
        close_reason: str,
        realized_pnl: float,
        close_price: float,
    ) -> Optional[PaperTrade]:
        stmt = (
            update(PaperTrade)
            .where(
                and_(
                    PaperTrade.id == trade_id,
                    PaperTrade.status == TradeStatus.OPEN,
                )
            )
            .values(
                status=TradeStatus.CLOSED,
                close_reason=close_reason,
                realized_pnl=realized_pnl,
                current_price=close_price,
                closed_at=datetime.now(tz=timezone.utc),
                updated_at=datetime.now(tz=timezone.utc),
            )
            .returning(PaperTrade)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # ── ANALYTICS ──────────────────────────────────────────────────────────────

    async def get_performance_summary(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
    ) -> dict:
        """
        Summary statistik performa paper trading.
        Digunakan oleh API /trades/performance endpoint.
        """
        since = datetime.now(tz=timezone.utc) - timedelta(days=days)

        base_filter = and_(
            PaperTrade.status == TradeStatus.CLOSED,
            PaperTrade.closed_at >= since,
        )
        if symbol:
            base_filter = and_(base_filter, PaperTrade.symbol == symbol.upper())

        # Total trades & Win/Loss count
        total_stmt = select(func.count(PaperTrade.id)).where(base_filter)
        total = (await self._session.execute(total_stmt)).scalar_one()

        win_stmt = select(func.count(PaperTrade.id)).where(
            and_(base_filter, PaperTrade.realized_pnl > 0)
        )
        wins = (await self._session.execute(win_stmt)).scalar_one()

        # PnL aggregates
        pnl_stmt = select(
            func.sum(PaperTrade.realized_pnl).label("total_pnl"),
            func.avg(PaperTrade.realized_pnl).label("avg_pnl"),
            func.max(PaperTrade.realized_pnl).label("best_trade"),
            func.min(PaperTrade.realized_pnl).label("worst_trade"),
        ).where(base_filter)
        pnl_result = (await self._session.execute(pnl_stmt)).one()

        # Action breakdown
        long_stmt = select(func.count(PaperTrade.id)).where(
            and_(base_filter, PaperTrade.action == "LONG")
        )
        longs = (await self._session.execute(long_stmt)).scalar_one()

        # Close reason breakdown
        tp_stmt = select(func.count(PaperTrade.id)).where(
            and_(base_filter, PaperTrade.close_reason == "TP_HIT")
        )
        tp_hits = (await self._session.execute(tp_stmt)).scalar_one()

        losses = total - wins
        win_rate = (wins / total * 100) if total > 0 else 0.0
        profit_factor = (
            abs(pnl_result.best_trade or 0) / abs(pnl_result.worst_trade or 1)
            if (pnl_result.worst_trade or 0) != 0
            else 0.0
        )

        return {
            "period_days": days,
            "symbol_filter": symbol or "ALL",
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usdt": round(pnl_result.total_pnl or 0, 2),
            "avg_pnl_usdt": round(pnl_result.avg_pnl or 0, 2),
            "best_trade_usdt": round(pnl_result.best_trade or 0, 2),
            "worst_trade_usdt": round(pnl_result.worst_trade or 0, 2),
            "long_trades": longs,
            "short_trades": total - longs,
            "tp_hit_count": tp_hits,
            "sl_hit_count": total - tp_hits,
            "profit_factor": round(profit_factor, 2),
        }

    async def get_open_trades_count(self) -> int:
        stmt = select(func.count(PaperTrade.id)).where(
            PaperTrade.status == TradeStatus.OPEN
        )
        return (await self._session.execute(stmt)).scalar_one()

    async def get_total_unrealized_pnl(self) -> float:
        stmt = select(func.sum(PaperTrade.unrealized_pnl)).where(
            PaperTrade.status == TradeStatus.OPEN
        )
        result = (await self._session.execute(stmt)).scalar_one()
        return float(result or 0.0)