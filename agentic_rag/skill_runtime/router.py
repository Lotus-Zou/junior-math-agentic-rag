"""Deterministic risk/cost router for configured pipelines."""

from __future__ import annotations

from agentic_rag.domain.schemas import CurriculumSolveOutput


class SkillRouter:
    def choose(self, state: dict, branches: dict[str, str]) -> str:
        solved = state.get("curriculum_solve")
        if isinstance(solved, CurriculumSolveOutput) and solved.handled and "deterministic" in branches:
            return branches["deterministic"]
        rewritten = state.get("query_rewrite")
        if rewritten and getattr(rewritten, "missing_conditions", None) and "clarify" in branches:
            return branches["clarify"]
        if "rag" in branches:
            return branches["rag"]
        return next(iter(branches.values()))

