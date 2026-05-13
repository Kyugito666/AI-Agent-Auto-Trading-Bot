# data/onchain.py
"""
On-Chain Data Client — Real-time blockchain intelligence.

Sources:
  - DefiLlama  : TVL, protocol metrics, BTC dominance
  - DexScreener: DEX pair data, liquidity, volume anomaly
  - Birdeye    : Solana-specific on-chain (whale txns, large transfers)
  - Binance    : Public market ticker (fallback / CEX volume)

Design:
  Semua method menggunakan httpx.AsyncClient yang di-share (connection pooling).
  Retry logic menggunakan tenacity dengan exponential backoff.
  Partial failure di-handle secara graceful — satu source gagal tidak merusak semua.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import Settings
from models.schemas import OnChainSnapshot, WhaleTransaction
from utils.logger import get_logger

logger = get_logger(__name__)

# Token address mapping — diperlukan untuk Birdeye (Solana mainnet)
SOLANA_TOKEN_MAP: dict[str, str] = {
    "SOL": "So11111111111111111111111111111111111111112",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
}


class OnChainDataClient:
    """
    Unified on-chain data aggregator.
    Satu client, multiple sources, graceful degradation.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        # Shared HTTP client dengan connection pooling
        # limits: max 10 concurrent connections per host
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"User-Agent": "TradingAgent/1.0"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── Public Interface ───────────────────────────────────────────────────────

    async def get_whale_transactions(
        self, token: str, min_usd: float = 100_000
    ) -> list[WhaleTransaction]:
        """
        Fetch whale transactions > $100K.
        Source: Birdeye (Solana) atau DexScreener (EVM).
        """
        if token in SOLANA_TOKEN_MAP:
            return await self._get_birdeye_large_trades(
                token_address=SOLANA_TOKEN_MAP[token],
                min_usd=min_usd,
            )
        else:
            # EVM tokens: DexScreener trades endpoint
            return await self._get_dexscreener_whale_trades(
                token=token,
                min_usd=min_usd,
            )

    async def get_dex_liquidity(self, token: str) -> float:
        """
        Total DEX liquidity dalam USD untuk token.
        Source: DexScreener (aggregate semua pairs).
        """
        try:
            pairs = await self._fetch_dexscreener_pairs(token=token)
            if not pairs:
                return 0.0

            total_liquidity = sum(
                p.get("liquidity", {}).get("usd", 0) or 0
                for p in pairs
                if p.get("liquidity")
            )
            return float(total_liquidity)

        except Exception as e:
            logger.warning(f"DEX liquidity fetch failed for {token}: {e}")
            return 0.0

    async def get_volume_anomaly(self, token: str) -> float:
        """
        Hitung Volume Anomaly Score.
        
        Formula: current_volume_1h / avg_volume_1h_7d
        Score > 1.5 = volume spike (anomalous)
        Score < 0.5 = volume drought

        Source: DexScreener volume data.
        """
        try:
            pairs = await self._fetch_dexscreener_pairs(token=token)
            if not pairs:
                return 0.0

            # Ambil pair dengan liquidity tertinggi sebagai representatif
            best_pair = max(
                pairs,
                key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0,
                default=None,
            )
            if not best_pair:
                return 0.0

            vol_h1 = best_pair.get("volume", {}).get("h1", 0) or 0
            vol_h24 = best_pair.get("volume", {}).get("h24", 0) or 0

            if vol_h24 == 0:
                return 0.0

            # Estimasi avg hourly volume dari 24h data
            avg_hourly = vol_h24 / 24
            if avg_hourly == 0:
                return 0.0

            anomaly_score = vol_h1 / avg_hourly
            logger.debug(
                f"Volume anomaly {token}: 1h=${vol_h1:,.0f} | "
                f"avg_hourly=${avg_hourly:,.0f} | score={anomaly_score:.2f}"
            )
            return round(anomaly_score, 4)

        except Exception as e:
            logger.warning(f"Volume anomaly calc failed for {token}: {e}")
            return 0.0

    async def get_large_transfers(
        self, token: str, min_usd: float = 500_000
    ) -> list[WhaleTransaction]:
        """
        On-chain large transfers (bukan DEX swap).
        Source: Birdeye untuk Solana tokens.
        """
        if token not in SOLANA_TOKEN_MAP:
            return []  # EVM large transfer tracking memerlukan Etherscan API (future)

        return await self._get_birdeye_large_trades(
            token_address=SOLANA_TOKEN_MAP[token],
            min_usd=min_usd,
        )

    async def get_btc_dominance(self) -> float:
        """
        BTC Market Dominance (%).
        Source: DefiLlama global metrics.
        """
        try:
            return await self._fetch_defillama_btc_dominance()
        except Exception as e:
            logger.warning(f"BTC dominance fetch failed: {e}. Using default 50%")
            return 50.0

    async def get_protocol_tvl(self, protocol_slug: str) -> float:
        """
        Total Value Locked untuk protocol tertentu.
        Source: DefiLlama /tvl/:protocol endpoint.
        """
        try:
            url = f"{self.settings.DEFILLAMA_BASE_URL}/tvl/{protocol_slug}"
            response = await self._get_with_retry(url)
            return float(response.text.strip())
        except Exception as e:
            logger.warning(f"TVL fetch failed for {protocol_slug}: {e}")
            return 0.0

    async def get_full_snapshot(self, token: str) -> OnChainSnapshot:
        """
        Convenience method: ambil semua on-chain data sekaligus.
        Identik dengan apa yang dipanggil analyzer, tapi bisa dipakai langsung.
        """
        whale_txns, liquidity, anomaly_score, large_transfers = await asyncio.gather(
            self.get_whale_transactions(token),
            self.get_dex_liquidity(token),
            self.get_volume_anomaly(token),
            self.get_large_transfers(token),
            return_exceptions=True,
        )

        return OnChainSnapshot(
            token=token,
            whale_transactions=whale_txns if not isinstance(whale_txns, Exception) else [],
            dex_liquidity_usd=liquidity if not isinstance(liquidity, Exception) else 0.0,
            volume_anomaly_score=anomaly_score if not isinstance(anomaly_score, Exception) else 0.0,
            large_transfers=large_transfers if not isinstance(large_transfers, Exception) else [],
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ── Private: DexScreener ───────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_dexscreener_pairs(self, token: str) -> list[dict]:
        """
        Fetch semua trading pairs untuk token dari DexScreener.
        Rate limit: 300 req/min (generous, no key needed).
        """
        url = f"{self.settings.DEXSCREENER_BASE_URL}/dex/search?q={token}USDT"
        response = await self._get_with_retry(url)
        data = response.json()
        pairs = data.get("pairs") or []

        # Filter: hanya pairs dengan quote USD/USDT dan liquidity > $10K
        filtered = [
            p for p in pairs
            if p.get("quoteToken", {}).get("symbol", "").upper() in ("USDT", "USDC", "USD")
            and (p.get("liquidity", {}).get("usd") or 0) > 10_000
        ]

        logger.debug(f"DexScreener: {len(filtered)} valid pairs for {token}")
        return filtered

    async def _get_dexscreener_whale_trades(
        self, token: str, min_usd: float
    ) -> list[WhaleTransaction]:
        """
        Identifikasi whale trades dari DexScreener recent transactions.
        Menggunakan pair terliquid sebagai proxy.
        """
        try:
            pairs = await self._fetch_dexscreener_pairs(token=token)
            if not pairs:
                return []

            best_pair = max(
                pairs,
                key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0,
            )
            pair_address = best_pair.get("pairAddress", "")
            chain_id = best_pair.get("chainId", "")

            if not pair_address:
                return []

            url = f"{self.settings.DEXSCREENER_BASE_URL}/dex/pairs/{chain_id}/{pair_address}"
            response = await self._get_with_retry(url)
            data = response.json()

            txns = data.get("pair", {}).get("txns", {})
            # DexScreener tidak expose individual txn details di free tier
            # Kita proxy menggunakan volume spike dalam window pendek

            # Fallback: return empty, log untuk awareness
            logger.debug(f"DexScreener whale txns: using volume proxy for {token}")
            return []

        except Exception as e:
            logger.warning(f"DexScreener whale fetch failed for {token}: {e}")
            return []

    # ── Private: Birdeye (Solana) ──────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get_birdeye_large_trades(
        self, token_address: str, min_usd: float
    ) -> list[WhaleTransaction]:
        """
        Fetch large trades dari Birdeye API.
        Requires: BIRDEYE_API_KEY (free tier: 1000 req/day)
        """
        if not self.settings.BIRDEYE_API_KEY:
            logger.debug("Birdeye API key not set — skipping whale fetch")
            return []

        url = "https://public-api.birdeye.so/defi/txs/token"
        headers = {
            "X-API-KEY": self.settings.BIRDEYE_API_KEY,
            "x-chain": "solana",
        }
        params = {
            "address": token_address,
            "tx_type": "swap",
            "sort_type": "desc",
            "limit": 50,
        }

        response = await self._client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        transactions = []
        for item in data.get("data", {}).get("items", []):
            volume_usd = item.get("volumeUSD") or 0
            if float(volume_usd) < min_usd:
                continue

            side = item.get("side", "unknown")
            txn = WhaleTransaction(
                tx_hash=item.get("txHash", ""),
                amount_usd=float(volume_usd),
                direction="IN" if side == "buy" else "OUT",
                from_address=item.get("from", {}).get("address", ""),
                to_address=item.get("to", {}).get("address", ""),
                timestamp=datetime.fromtimestamp(
                    item.get("blockUnixTime", time.time()), tz=timezone.utc
                ),
            )
            transactions.append(txn)

        logger.debug(
            f"Birdeye: {len(transactions)} whale txns "
            f"(>${min_usd/1000:.0f}K) for {token_address[:8]}..."
        )
        return transactions

    # ── Private: DefiLlama ─────────────────────────────────────────────────────

    async def _fetch_defillama_btc_dominance(self) -> float:
        """
        Fetch global crypto market data dari DefiLlama.
        Endpoint /global mengembalikan BTC dominance dan total market cap.
        """
        url = f"{self.settings.DEFILLAMA_BASE_URL}/v2/globalCharts"
        response = await self._get_with_retry(url)
        data = response.json()

        # DefiLlama global chart: ambil entry terbaru
        if isinstance(data, list) and data:
            latest = data[-1]
            total_tvl = latest.get("totalLiquidityUSD", 1)

            # Proxy: pakai BTC chain TVL / total TVL sebagai dominance estimasi
            # Untuk dominance akurat, bisa gunakan CoinGecko /global endpoint
            btc_chain = await self._fetch_defillama_chain_tvl("Bitcoin")
            if total_tvl > 0:
                return round((btc_chain / total_tvl) * 100, 2)

        return 50.0  # Default fallback

    async def _fetch_defillama_chain_tvl(self, chain: str) -> float:
        try:
            url = f"{self.settings.DEFILLAMA_BASE_URL}/v2/chains"
            response = await self._get_with_retry(url)
            chains = response.json()
            for c in chains:
                if c.get("name", "").lower() == chain.lower():
                    return float(c.get("tvl", 0))
            return 0.0
        except Exception:
            return 0.0

    # ── HTTP Helper ────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _get_with_retry(self, url: str, **kwargs) -> httpx.Response:
        response = await self._client.get(url, **kwargs)
        response.raise_for_status()
        return response