"""
Redis result cache (cache-aside pattern).

The search pipeline is expensive: embed the query, FAISS search, BM25 scan,
then a cross-encoder rerank of ~30 docs. That's hundreds of ms (seconds on a
cold GPU alloc). But the *same query* always produces the *same results* until
the index changes — so we cache the final result list in Redis keyed by query.

Design decisions worth knowing (interview-ready):

  * Cache-aside (a.k.a. lazy loading): the app checks the cache first; on a miss
    it computes, then writes back. Redis never talks to the pipeline directly.
  * TTL: every entry auto-expires (CACHE_TTL). This bounds staleness — if the
    index is rebuilt, stale entries die on their own within the window.
  * Key versioning: keys are prefixed `search:v1:`. Bump the version to
    invalidate the ENTIRE cache instantly (e.g. after re-indexing) without a
    FLUSHDB.
  * Key normalization: "Rust " and "rust" hit the same entry.
  * Graceful degradation: the cache is an OPTIMIZATION, never a dependency. If
    Redis is unset, unreachable, or errors, search still works — we just skip
    the cache. Every Redis call is wrapped so a cache fault can't break search.
  * JSON serialization: portable and human-readable in redis-cli / the Upstash
    console. (pickle is faster but opaque and unsafe to load from a shared store.)
  * Atomic INCR counters track hit/miss so you can prove the cache is working.

Enable by setting REDIS_URL (e.g. an Upstash `rediss://...` URL). Unset = no cache.
"""

import os
import json

try:
    import redis  # redis-py: the standard Python Redis client
except ImportError:  # dependency optional — search runs fine without it
    redis = None

REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_TTL = int(os.environ.get("CACHE_TTL", 86400))  # seconds; 24h default
KEY_PREFIX = "search:v1:"                             # bump v1 -> v2 to invalidate all
HITS_KEY = "search:stats:hits"
MISSES_KEY = "search:stats:misses"

# Lazy singleton. We connect once on first use, not at import, so a slow or
# missing Redis never blocks server boot. `_init_done` ensures we only *attempt*
# the connection once — after a failure we stay in no-cache mode for the process.
_client = None
_init_done = False


def _client_or_none():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True

    if not REDIS_URL or redis is None:
        return None  # caching disabled — this is a normal, supported mode

    try:
        # from_url builds a connection pool. Short timeouts so a dead Redis
        # degrades to "no cache" fast instead of hanging every request.
        client = redis.from_url(
            REDIS_URL,
            decode_responses=True,      # GET returns str, not bytes -> json.loads works
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()                   # verify the connection actually works
        _client = client
        print("[cache] Redis connected — result caching ON")
    except Exception as e:
        print(f"[cache] Redis unavailable ({e}) — running WITHOUT cache")
        _client = None
    return _client


def _key(query: str) -> str:
    return KEY_PREFIX + query.strip().lower()


def cache_get(query: str):
    """Return cached results for `query`, or None on miss / cache unavailable."""
    client = _client_or_none()
    if client is None:
        return None
    try:
        raw = client.get(_key(query))
        if raw is None:
            client.incr(MISSES_KEY)     # atomic counter — no race even under load
            return None
        client.incr(HITS_KEY)
        return json.loads(raw)
    except Exception as e:
        print(f"[cache] get failed ({e}) — treating as miss")
        return None


def cache_set(query: str, results) -> None:
    """Store `results` for `query` with a TTL. Silent no-op if cache is off."""
    client = _client_or_none()
    if client is None:
        return
    try:
        # SETEX = set value + expiry atomically (one round-trip).
        client.setex(_key(query), CACHE_TTL, json.dumps(results))
    except Exception as e:
        print(f"[cache] set failed ({e}) — skipping write")


def cache_stats() -> dict:
    """Hit/miss counters — handy for a /stats endpoint or logs."""
    client = _client_or_none()
    if client is None:
        return {"enabled": False}
    try:
        hits = int(client.get(HITS_KEY) or 0)
        misses = int(client.get(MISSES_KEY) or 0)
        total = hits + misses
        return {
            "enabled": True,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 3) if total else 0.0,
        }
    except Exception:
        return {"enabled": True, "error": "stats unavailable"}
