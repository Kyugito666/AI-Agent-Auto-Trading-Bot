# api/routes/agent.py
"""
Agent REST API Routes.

Endpoints:
  POST /api/v1/agent/analyze      → Trigger analisis manual untuk satu symbol
  POST /api/v1/agent/run-cycle    → Trigger full cycle untuk semua pairs
  GET  /api/v1/agent/status       → Status sistem & last analysis
  GET  /api/v1/agent/signals      → History sinyal yang dihasilkan
"""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


# ── Request / Response Schemas ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    symbol: str = Field(
        default="BTCUSDT",
        description="Trading pair to analyze",
        examples=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    force_refresh: bool = Field(
        default=False,
        description="Bypass cache dan paksa fetch data baru",
    )


class SignalResponse(BaseModel):
    symbol: str
    action: str                    # LONG / SHORT / HOLD
    confidence: float
    entry_price: float
    take_profit: float
    stop_loss: float
    risk_reward_ratio: float
    sentiment_score: float
    btc_trend: Optional[str]
    reasoning: str
    llm_provider: str
    timestamp: datetime
    trade_id: Optional[str]        # Set jika paper trade dibuat
    error: Optional[str]


class SystemStatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    scheduler_running: bool
    supported_pairs: list[str]
    open_paper_trades: int
    total_unrealized_pnl: float
    last_cycle_at: Optional[datetime]
    paper_capital: float


# ── In-memory state (lightweight, tidak perlu DB) ─────────────────────────────
_last_signals: dict[str, SignalResponse] = {}
_system_start_time = datetime.now(tz=timezone.utc)
_last_cycle_at: Optional[datetime] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=SignalResponse)
async def analyze_symbol(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger analisis manual untuk satu symbol.
    Menjalankan full pipeline secara synchronous dan mengembalikan hasilnya.

    Note: Endpoint ini blocking. Untuk production volume tinggi,
    pertimbangkan async job queue (Celery/ARQ).
    """
    from main import app_state

    symbol = request.symbol.upper()

    if symbol not in settings.SUPPORTED_PAIRS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Symbol '{symbol}' tidak didukung. "
                f"Supported: {settings.SUPPORTED_PAIRS}"
            ),
        )

    if not hasattr(app_state, "analyzer") or app_state.analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Analyzer belum siap. Tunggu beberapa detik setelah startup.",
        )

    logger.info(f"Manual analysis triggered for {symbol}")

    result = await app_state.analyzer.run_full_pipeline(symbol=symbol)

    # Build response
    response = SignalResponse(
        symbol=result.symbol,
        action=result.signal.action.value,
        confidence=result.signal.confidence,
        entry_price=result.signal.entry_price,
        take_profit=result.signal.take_profit,
        stop_loss=result.signal.stop_loss,
        risk_reward_ratio=result.signal.risk_reward_ratio,
        sentiment_score=result.sentiment_score,
        btc_trend=result.btc_state.trend if result.btc_state else None,
        reasoning=result.signal.reasoning,
        llm_provider=result.llm_analysis.provider_used,
        timestamp=result.timestamp,
        trade_id=result.trade_id,
        error=result.error,
    )

    # Cache signal terbaru
    _last_signals[symbol] = response
    return response


@router.post("/run-cycle")
async def run_full_cycle(background_tasks: BackgroundTasks):
    """
    Trigger full analysis cycle untuk semua SUPPORTED_PAIRS.
    Dijalankan sebagai background task agar endpoint langsung return.
    """
    from main import app_state, _run_analysis_cycle

    global _last_cycle_at

    logger.info("Manual full cycle triggered via API")
    _last_cycle_at = datetime.now(tz=timezone.utc)

    # Jalankan di background — tidak blocking response
    background_tasks.add_task(_run_analysis_cycle)

    return {
        "status": "accepted",
        "message": f"Full cycle started for {len(settings.SUPPORTED_PAIRS)} pairs",
        "pairs": settings.SUPPORTED_PAIRS,
        "triggered_at": _last_cycle_at.isoformat(),
    }


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Status sistem real-time: scheduler, open trades, capital.
    """
    from main import app_state
    from repositories.trade_repository import TradeRepository
    from models.database import AsyncSessionLocal

    uptime = (datetime.now(tz=timezone.utc) - _system_start_time).total_seconds()

    # Query DB untuk trade metrics
    open_trades = 0
    unrealized_pnl = 0.0

    try:
        async with AsyncSessionLocal() as session:
            repo = TradeRepository(session)
            open_trades = await repo.get_open_trades_count()
            unrealized_pnl = await repo.get_total_unrealized_pnl()
    except Exception as e:
        logger.warning(f"DB query failed in status endpoint: {e}")

    return SystemStatusResponse(
        status="operational",
        uptime_seconds=round(uptime, 1),
        scheduler_running=(
            hasattr(app_state, "scheduler") and app_state.scheduler.running
        ),
        supported_pairs=settings.SUPPORTED_PAIRS,
        open_paper_trades=open_trades,
        total_unrealized_pnl=round(unrealized_pnl, 2),
        last_cycle_at=_last_cycle_at,
        paper_capital=settings.PAPER_CAPITAL,
    )


@router.get("/signals", response_model=list[SignalResponse])
async def get_last_signals():
    """
    Kembalikan sinyal terbaru untuk setiap pair yang sudah dianalisis.
    """
    return list(_last_signals.values())


@router.get("/signals/{symbol}", response_model=SignalResponse)
async def get_signal_for_symbol(symbol: str):
    """Sinyal terakhir untuk symbol tertentu."""
    sym = symbol.upper()
    if sym not in _last_signals:
        raise HTTPException(
            status_code=404,
            detail=f"Belum ada sinyal untuk {sym}. Trigger /analyze terlebih dahulu.",
        )
    return _last_signals[sym]