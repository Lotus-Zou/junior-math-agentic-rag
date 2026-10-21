# -*- coding: utf-8 -*-
"""Redis-backed answer cache with an in-process TTL fallback."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from threading import Lock
from cachetools import TTLCache

from config import (
    ANSWER_CACHE_VERSION, CACHE_LOCAL_TTL_SECONDS, CACHE_TTL_SECONDS,
    CRITIC_MODEL_NAME, KNOWLEDGE_INDEX_VERSION, LLM_MODEL_NAME, REDIS_URL,
)

logger = logging.getLogger(__name__)


def _digest(prefix: str, payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self):
        self.local = TTLCache(maxsize=1024, ttl=CACHE_LOCAL_TTL_SECONDS)
        self.lock = Lock()
        self.redis = None
        if REDIS_URL:
            try:
                import redis
                self.redis = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=.3, socket_timeout=.3)
            except Exception:
                self.redis = None

    @staticmethod
    def key(payload: dict) -> str:
        return _digest("math-rag:", {
            "request": payload, "knowledge_version": KNOWLEDGE_INDEX_VERSION,
            "answer_version": ANSWER_CACHE_VERSION, "generator": LLM_MODEL_NAME,
            "critic": CRITIC_MODEL_NAME,
        })

    def get(self, payload: dict):
        key = self.key(payload)
        with self.lock:
            value = self.local.get(key)
            if value is not None:
                return deepcopy(value)
        if self.redis:
            try:
                raw = self.redis.get(key)
                value = json.loads(raw) if raw else None
                if value is not None:
                    with self.lock:
                        self.local[key] = deepcopy(value)
                return value
            except Exception:
                logger.warning("shared answer cache unavailable", exc_info=True)
        return None

    def set(self, payload: dict, value: dict):
        key = self.key(payload)
        with self.lock:
            self.local[key] = deepcopy(value)
        if self.redis:
            try:
                self.redis.setex(key, CACHE_TTL_SECONDS, json.dumps(value, ensure_ascii=False))
            except Exception:
                logger.warning("shared answer cache write failed", exc_info=True)


answer_cache = AnswerCache()


class JsonTTLCache:
    """Shared Redis cache with bounded local fallback for immutable read results."""

    def __init__(self, prefix: str, maxsize: int = 2048):
        self.prefix = prefix
        self.local = TTLCache(maxsize=maxsize, ttl=CACHE_LOCAL_TTL_SECONDS)
        self.lock = Lock()
        self.redis = answer_cache.redis

    def key(self, payload: dict) -> str:
        return _digest(self.prefix, payload)

    def get(self, payload: dict):
        key = self.key(payload)
        with self.lock:
            value = self.local.get(key)
            if value is not None:
                return deepcopy(value)
        if self.redis:
            try:
                raw = self.redis.get(key)
                value = json.loads(raw) if raw else None
                if value is not None:
                    with self.lock:
                        self.local[key] = deepcopy(value)
                return value
            except Exception:
                logger.warning("shared retrieval cache unavailable", exc_info=True)
        return None

    def set(self, payload: dict, value: dict) -> None:
        key = self.key(payload)
        encoded = json.dumps(value, ensure_ascii=False)
        with self.lock:
            self.local[key] = deepcopy(value)
        if self.redis:
            try:
                self.redis.setex(key, CACHE_TTL_SECONDS, encoded)
            except Exception:
                logger.warning("shared retrieval cache write failed", exc_info=True)

    def delete(self, payload: dict) -> None:
        key = self.key(payload)
        with self.lock:
            self.local.pop(key, None)
        if self.redis:
            self.redis.delete(key)


retrieval_cache = JsonTTLCache("math-rag:retrieval:")
