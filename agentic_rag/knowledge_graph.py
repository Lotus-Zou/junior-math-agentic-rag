# -*- coding: utf-8 -*-
"""Lightweight GraphRAG for junior-high mathematics prerequisite expansion."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Iterable, List

from agentic_rag.math_taxonomy import PREREQUISITES
from config import KNOWLEDGE_GRAPH_PATH


class MathKnowledgeGraph:
    """A small, auditable knowledge-point dependency graph."""

    def __init__(self, path: str = KNOWLEDGE_GRAPH_PATH):
        self.path = Path(path)
        self._lock = RLock()
        self._edges = {point: list(required) for point, required in PREREQUISITES.items()}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            for point, required in payload.get("prerequisites", {}).items():
                self._edges[point] = list(dict.fromkeys(required))
        except (OSError, ValueError, TypeError):
            return

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "prerequisites": self._edges}
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, knowledge_point: str, prerequisites: Iterable[str]) -> None:
        if not knowledge_point:
            return
        with self._lock:
            existing = self._edges.setdefault(knowledge_point, [])
            self._edges[knowledge_point] = list(dict.fromkeys([*existing, *filter(None, prerequisites)]))

    def match_nodes(self, seeds: Iterable[str]) -> List[str]:
        matches = []
        for seed in seeds:
            for node in self._edges:
                if seed in node or node in seed:
                    matches.append(node)
        return list(dict.fromkeys(matches))

    def expand(self, seeds: Iterable[str], depth: int = 1) -> List[str]:
        frontier = self.match_nodes(seeds)
        expanded = []
        for _ in range(max(0, depth)):
            next_frontier = []
            for node in frontier:
                for prerequisite in self._edges.get(node, []):
                    if prerequisite not in expanded:
                        expanded.append(prerequisite)
                        next_frontier.extend(self.match_nodes([prerequisite]))
            frontier = list(dict.fromkeys(next_frontier))
        return expanded

    def context(self, seeds: Iterable[str]) -> str:
        nodes = self.match_nodes(seeds)
        if not nodes:
            return "无匹配的知识依赖关系。"
        return "\n".join(
            f"{node} -> 前置知识点：{'、'.join(self._edges.get(node, [])) or '无'}"
            for node in nodes
        )

    def as_dict(self) -> dict:
        return {"prerequisites": dict(self._edges)}


math_knowledge_graph = MathKnowledgeGraph()
