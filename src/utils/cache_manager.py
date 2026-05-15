"""
EMMDS Cache Manager
Avoids recomputing pipeline results for the same dataset.
Uses MD5 hash of the CSV content as the cache key.
Stores results in cache/ as JSON.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("cache")


def _default(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    return str(obj)


class CacheManager:
    """
    Transparent caching layer for the EMMDS pipeline.

    Cache key = MD5(dataset_csv + target_col + scaler).
    If a cached result exists and is not stale, return it immediately
    without re-running the entire pipeline.
    """

    def __init__(
        self,
        cache_dir: str | Path = CACHE_DIR,
        ttl_seconds: int = 86_400,          # 24-hour default TTL
        enabled: bool = True,
    ):
        self.cache_dir   = Path(cache_dir)
        self.ttl         = ttl_seconds
        self.enabled     = enabled
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits  = 0
        self._misses = 0

    # ── Public API ────────────────────────────────────────────────

    def dataset_hash(
        self,
        df: pd.DataFrame,
        target_col: str = "",
        scaler: str = "",
        extra: str = "",
    ) -> str:
        """
        Stable hash from dataset content + run parameters.
        Changing any parameter invalidates the cache.
        """
        raw = df.to_csv(index=False) + target_col + scaler + extra
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        """
        Return cached result for key, or None if miss/stale/disabled.
        """
        if not self.enabled:
            return None

        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(path.read_text())
            age  = time.time() - data.get("_cached_at", 0)
            if age > self.ttl:
                logger.info(f"Cache expired for key={key[:8]}… (age={age:.0f}s)")
                path.unlink(missing_ok=True)
                self._misses += 1
                return None

            self._hits += 1
            logger.info(f"Cache HIT  key={key[:8]}… (age={age:.0f}s)")
            return data.get("payload")

        except Exception as e:
            logger.warning(f"Cache read error for {key[:8]}…: {e}")
            self._misses += 1
            return None

    def set(self, key: str, payload: dict) -> None:
        """Store a result payload under key."""
        if not self.enabled:
            return

        # Strip non-serialisable objects (trained models, arrays)
        safe = self._make_serialisable(payload)
        record = {"_cached_at": time.time(), "key": key, "payload": safe}

        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(record, default=_default, indent=2))
            logger.info(f"Cache SET  key={key[:8]}… → {path}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def invalidate(self, key: str) -> bool:
        """Delete a specific cache entry."""
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Cache invalidated: {key[:8]}…")
            return True
        return False

    def clear_all(self) -> int:
        """Delete all cached results. Returns number of entries removed."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            count += 1
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    def stats(self) -> dict:
        """Return cache hit/miss stats."""
        total = self._hits + self._misses
        return {
            "hits":       self._hits,
            "misses":     self._misses,
            "total":      total,
            "hit_rate":   round(self._hits / total, 3) if total > 0 else 0.0,
            "cache_size": len(list(self.cache_dir.glob("*.json"))),
            "enabled":    self.enabled,
        }

    def list_entries(self) -> list:
        """List all cached runs with metadata."""
        entries = []
        for p in sorted(self.cache_dir.glob("*.json"), reverse=True):
            try:
                data  = json.loads(p.read_text())
                age   = time.time() - data.get("_cached_at", 0)
                payload = data.get("payload", {})
                entries.append({
                    "key":        p.stem,
                    "age_hours":  round(age / 3600, 1),
                    "best_model": payload.get("decision", {}).get("best_model"),
                    "trust":      payload.get("decision", {}).get("trust_score"),
                })
            except Exception:
                pass
        return entries

    # ── Internal ──────────────────────────────────────────────────

    def _make_serialisable(self, obj: Any) -> Any:
        """Recursively strip non-JSON-serialisable objects."""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k.startswith("_"):          # Skip private keys (models, preprocessors)
                    continue
                result[k] = self._make_serialisable(v)
            return result
        if isinstance(obj, list):
            return [self._make_serialisable(i) for i in obj]
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, pd.DataFrame):    return obj.head(5).to_dict(orient="records")
        if callable(obj):                    return None
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)
