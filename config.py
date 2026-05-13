# config.py
"""
Centralized configuration using pydantic-settings.
Semua secrets WAJIB berasal dari environment variables / .env file.
TIDAK ADA hardcoded credentials di source code.
"""
from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "AI Trading Agent"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # ── LLM Providers ─────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(..., description="Primary LLM provider")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"  # Best free-tier reasoning model
    GROQ_RPM_LIMIT: int = 30  # Requests per minute (free tier)

    GEMINI_API_KEY: str = Field(..., description="Fallback LLM provider")
    GEMINI_MODEL: str = "gemini-2.0-flash"

    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.1  # Low temp untuk trading — deterministik

    # ── On-Chain & Market Data ─────────────────────────────────────────────
    BIRDEYE_API_KEY: str = Field(default="", description="Birdeye (Solana on-chain)")
    DEXSCREENER_BASE_URL: str = "https://api.dexscreener.com/latest"
    DEFILLAMA_BASE_URL: str = "https://api.llama.fi"
    BINANCE_BASE_URL: str = "https://api.binance.com"  # Public endpoint, no key needed

    # ── News & Sentiment ──────────────────────────────────────────────────
    CRYPTOPANIC_API_KEY: str = Field(default="", description="CryptoPanic news feed")
    CRYPTOPANIC_BASE_URL: str = "https://cryptopanic.com/api/v1"

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./trading_agent.db"
    # Untuk prod: "postgresql+asyncpg://user:pass@localhost/trading_db"

    # ── Trading Parameters ────────────────────────────────────────────────
    DEFAULT_PAIR: str = "BTCUSDT"
    SUPPORTED_PAIRS: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

    # Risk Management Defaults
    DEFAULT_LEVERAGE: int = 1          # Paper trading: no leverage default
    DEFAULT_RISK_PCT: float = 0.02     # 2% risk per trade
    DEFAULT_RR_RATIO: float = 2.0      # Risk:Reward 1:2
    PAPER_CAPITAL: float = 10_000.0    # Starting paper capital (USDT)

    # Pipeline timing
    ANALYSIS_INTERVAL_SECONDS: int = 60   # Jalankan full cycle setiap 60 detik
    ONCHAIN_REFRESH_SECONDS: int = 30
    NEWS_REFRESH_SECONDS: int = 120

    @field_validator("LLM_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature harus antara 0.0 dan 2.0")
        return v


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance. Cached setelah pertama kali diload."""
    return Settings()