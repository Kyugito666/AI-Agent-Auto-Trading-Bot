# api/routes/trades.py
"""
Trade History & Performance API Routes.

Endpoints:
  GET /api/v1/trades/open              → Semua open paper trades
  GET /api/v1/trades/closed            → History trades yang sudah closed
  GET /api/v1/trades/performance       → Statistik performa (win rate, PnL, dll)
  GET /api/v1/trades/{trade_id}        → Detail satu trade
  PATCH /api/v1/trades/{trade_id}/close → Manual close trade (paper)
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from models.database import get_db
from models.orm_models import PaperTrade
from repositories.trade_repository import TradeRepository
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# ── Response Schemas ───────────────────────────────────────────────────────────

class TradeResponse(BaseModel):
    id: str
    symbol: str
    action: str
    status: str
    entry_price: float
    take_profit: float
    stop_loss: float
    quantity: float
    notional_value: float
    risk_amount: float
    leverage: int
    confidence: float
    sentiment_score: Optional[float]
    btc_trend: Optional[str]
    llm_provider: Optional[str]
    current_price: Optional[float]
    unrealized_pnl: Optional[float]
    realized_pnl: Optional[float]
    close_reason: Optional[str]
    capital_before: float
    signal_reasoning: Optional[str]
    opened_at: str
    closed_at: Optional[str]

    @classmethod
    def from_orm(cls, trade: PaperTrade) -> "TradeResponse":
        return cls(
            id=trade.id,
            symbol=trade.symbol,
            action=trade.action,
            status=trade.status.value,
            entry_price=trade.entry_price,
            take_profit=trade.take_profit,
            stop_loss=trade.stop_loss,
            quantity=trade.quantity,
            notional_value=trade.notional_value,
            risk_amount=trade.risk_amount,
            leverage=trade.leverage,
            confidence=trade.confidence,
            sentiment_score=trade.sentiment_score,
            btc_trend=trade.btc_trend,
            llm_provider=trade.llm_provider,
            current_price=trade.current_price,
            unrealized_pnl=trade.unrealized_pnl,
            realized_pnl=trade.realized_pnl,
            close_reason=trade.close_reason,
            capital_before=trade.capital_before,
            signal_reasoning=trade.signal_reasoning,
            opened_at=trade.opened_at.isoformat(),
            closed_at=trade.closed_at.isoformat() if trade.closed_at else None,
        )


class PerformanceResponse(BaseModel):
    period_days: int
    symbol_filter: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_pnl_usdt: float
    avg_pnl_usdt: float
    best_trade_usdt: float
    worst_trade_usdt: float
    long_trades: int
    short_trades: int
    tp_hit_count: int
    sl_hit_count: int
    profit_factor: float


class ManualCloseRequest(BaseModel):
    close_price: float
    reason: str = "MANUAL"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/open", response_model=list[TradeResponse])
async def get_open_trades(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    db: AsyncSession = Depends(get_db),
):
    """Semua paper trades yang sedang OPEN."""
    repo = TradeRepository(db)
    trades = await repo.list_open_trades(symbol=symbol)
    return [TradeResponse.from_orm(t) for t in trades]


@router.get("/closed", response_model=list[TradeResponse])
async def get_closed_trades(
    symbol: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """History paper trades yang sudah CLOSED, dengan pagination."""
    repo = TradeRepository(db)
    trades = await repo.list_closed_trades(symbol=symbol, limit=limit, offset=offset)
    return [TradeResponse.from_orm(t) for t in trades]


@router.get("/recent", response_model=list[TradeResponse])
async def get_recent_trades(
    hours: int = Query(default=24, ge=1, le=168, description="Lookback window dalam jam"),
    db: AsyncSession = Depends(get_db),
):
    """Trades dari N jam terakhir (default: 24 jam)."""
    repo = TradeRepository(db)
    trades = await repo.list_recent(hours=hours)
    return [TradeResponse.from_orm(t) for t in trades]


@router.get("/performance", response_model=PerformanceResponse)
async def get_performance(
    symbol: Optional[str] = Query(default=None, description="Filter by symbol"),
    days: int = Query(default=30, ge=1, le=365, description="Periode analisis dalam hari"),
    db: AsyncSession = Depends(get_db),
):
    """
    Statistik performa paper trading agent.
    Termasuk: win rate, total PnL, profit factor, breakdown TP vs SL.
    """
    repo = TradeRepository(db)
    summary = await repo.get_performance_summary(symbol=symbol, days=days)
    return PerformanceResponse(**summary)


@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade_detail(
    trade_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Detail satu paper trade by ID."""
    repo = TradeRepository(db)
    trade = await repo.get_by_id(trade_id)

    if not trade:
        raise HTTPException(
            status_code=404,
            detail=f"Trade '{trade_id}' tidak ditemukan",
        )
    return TradeResponse.from_orm(trade)


@router.patch("/{trade_id}/close", response_model=TradeResponse)
async def manual_close_trade(
    trade_id: str,
    body: ManualCloseRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Manual close paper trade pada harga tertentu.
    Berguna untuk evaluasi atau testing skenario.
    """
    repo = TradeRepository(db)
    trade = await repo.get_by_id(trade_id)

    if not trade:
        raise HTTPException(status_code=404, detail=f"Trade '{trade_id}' tidak ditemukan")

    if trade.status.value != "OPEN":
        raise HTTPException(
            status_code=400,
            detail=f"Trade sudah {trade.status.value}, tidak bisa di-close lagi",
        )

    # Hitung realized PnL
    if trade.action == "LONG":
        realized_pnl = (body.close_price - trade.entry_price) * trade.quantity
    else:
        realized_pnl = (trade.entry_price - body.close_price) * trade.quantity

    async with db.begin():
        updated = await repo.close_trade(
            trade_id=trade_id,
            close_reason=body.reason,
            realized_pnl=realized_pnl,
            close_price=body.close_price,
        )

    if not updated:
        raise HTTPException(status_code=500, detail="Gagal menutup trade")

    logger.info(
        f"Manual close | {trade_id} | {body.reason} | "
        f"PnL: ${realized_pnl:+,.2f}"
    )
    return TradeResponse.from_orm(updated)