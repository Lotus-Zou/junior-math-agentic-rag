"""Central policy checks applied before Skill handlers run."""

from __future__ import annotations

from agentic_rag.skill_runtime.contracts import SkillContext
from agentic_rag.skill_runtime.errors import FatalSkillError
from agentic_rag.skill_runtime.manifest import SkillManifest


class PolicyEngine:
    def authorize(self, manifest: SkillManifest, context: SkillContext) -> list[str]:
        decisions = []
        if manifest.side_effects and not manifest.idempotent:
            required = f"allow:{manifest.id}"
            if required not in context.policy_set:
                raise FatalSkillError("Side effect permission denied", safe_message="当前操作未获授权。")
            decisions.append(f"{required}:pass")
        for policy in manifest.policies:
            decisions.append(f"{policy}:pass")
        return decisions

