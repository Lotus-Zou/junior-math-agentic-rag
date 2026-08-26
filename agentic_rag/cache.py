# -*- coding: utf-8 -*-
"""Redis-backed answer cache with an in-process TTL fallback."""

from __future__ import annotations

import hashlib
import json
from threading import Lock
from cachetools import TTLCache

from config import CACHE_TTL_SECONDS, REDIS_URL


class AnswerCache:
    def __init__(self):
        self.local = TTLCache(maxsize=1024, ttl=CACHE_TTL_SECONDS)
        self.lock = Lock()
        self.redis = None
        if REDIS_URL:
            try:
                import redis
                self.redis = redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=1)
                self.redis.ping()
            except Exception:
                self.redis = None

    @staticmethod
    def key(payload: dict) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return "math-rag:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, payload: dict):
        key = self.key(payload)
        if self.redis:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        with self.lock:
            return self.local.get(key)

    def set(self, payload: dict, value: dict):
        key = self.key(payload)
        if self.redis:
            self.redis.setex(key, CACHE_TTL_SECONDS, json.dumps(value, ensure_ascii=False))
        else:
            with self.lock:
                self.local[key] = value


answer_cache = AnswerCache()
