"""Deterministic risk/cost router for configured pipelines."""

from __future__ import annotations

from agentic_rag.domain.schemas import CurriculumSolveOutput, TurnRouteOutput


class SkillRouter:
    def choose(self, state: dict, branches: dict[str, str]) -> str:
        solved = state.get("curriculum_solve")
        if isinstance(solved, CurriculumSolveOutput):
            if solved.handled and "deterministic" in branches:
                return branches["deterministic"]
            if not solved.handled and "solve_agent" in branches:
                return branches["solve_agent"]
            if not solved.handled and "rag" in branches:
                return branches["rag"]
        turn = state.get("turn_router")
        if isinstance(turn, TurnRouteOutput) and turn.route in branches:
            return branches[turn.route]
        if isinstance(turn, TurnRouteOutput) and turn.intent in branches:
            return branches[turn.intent]
        rewritten = state.get("query_rewrite")
        if rewritten and getattr(rewritten, "missing_conditions", None) and "clarify" in branches:
            return branches["clarify"]
        if "rag" in branches:
            return branches["rag"]
        if "default" in branches:
            return branches["default"]
        return next(iter(branches.values()))
