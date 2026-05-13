# main.py
"""
Entry point sistem AI Trading Agent.
Bertanggung jawab untuk:
  1. Inisialisasi FastAPI application
  2. Bootstrap database (create tables)
  3. Mendaftarkan APScheduler jobs (pipeline loop)
  4. Registrasi routes dan middleware
"""
import asyncio
import contextlib
from collections.abc import AsyncIterator

import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import agent, trades
from config import get_settings
from core.analyzer import TradingAnalyzer
from models.database import engine, Base
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ── Application State ──────────────────────────────────────────────────────────
class AppState:
    """Menyimpan shared resources yang hidup sepanjang lifecycle aplikasi."""
    scheduler: AsyncIOScheduler
    analyzer: TradingAnalyzer


app_state = AppState()


# ── Lifespan Context Manager ───────────────────────────────────────────────────
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan handler menggantikan deprecated on_event('startup'/'shutdown').
    Semua resource dibuat di sini dan di-cleanup saat shutdown.
    """
    logger.info("🚀 Initializing AI Trading Agent System...")

    # 1. Bootstrap database
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database initialized")

    # 2. Inisialisasi core analyzer
    app_state.analyzer = TradingAnalyzer(settings=settings)
    await app_state.analyzer.initialize()
    logger.info("✅ TradingAnalyzer ready")

    # 3. Setup & start scheduler
    app_state.scheduler = AsyncIOScheduler(timezone="UTC")

    app_state.scheduler.add_job(
        func=_run_analysis_cycle,
        trigger=IntervalTrigger(seconds=settings.ANALYSIS_INTERVAL_SECONDS),
        id="main_analysis_cycle",
        name="Full Analysis Pipeline",
        replace_existing=True,
        misfire_grace_time=30,
    )

    app_state.scheduler.start()
    logger.info(
        f"✅ Scheduler started — analysis cycle every {settings.ANALYSIS_INTERVAL_SECONDS}s"
    )

    # Jalankan satu cycle pertama segera setelah startup
    asyncio.create_task(_run_analysis_cycle())

    yield  # ← Aplikasi berjalan di sini

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("🛑 Shutting down AI Trading Agent...")
    app_state.scheduler.shutdown(wait=False)
    await engine.dispose()
    logger.info("✅ Clean shutdown complete")


async def _run_analysis_cycle() -> None:
    """
    Wrapper untuk scheduled job. Error di sini tidak boleh crash scheduler.
    """
    try:
        logger.info("⏱️  Starting scheduled analysis cycle...")
        for pair in settings.SUPPORTED_PAIRS:
            await app_state.analyzer.run_full_pipeline(symbol=pair)
    except Exception as e:
        logger.exception(f"❌ Analysis cycle failed: {e}")


# ── FastAPI App Factory ────────────────────────────────────────────────────────
def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="AI-powered crypto trading agent with paper trading mode",
        lifespan=lifespan,
        docs_url="/docs" if settings.APP_ENV != "production" else None,
    )

    # Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Sesuaikan saat deploy
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    application.include_router(agent.router, prefix="/api/v1/agent", tags=["Agent"])
    application.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])

    @application.get("/health")
    async def health_check():
        return {
            "status": "operational",
            "env": settings.APP_ENV,
            "scheduler_running": app_state.scheduler.running,
        }

    return application


app = create_application()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.APP_ENV == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )