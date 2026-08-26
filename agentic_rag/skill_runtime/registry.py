"""Version-aware discovery and dependency validation for business Skills."""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from agentic_rag.skill_runtime.errors import ManifestError
from agentic_rag.skill_runtime.manifest import SkillManifest


class SkillRegistry:
    def __init__(self):
        self._items: dict[str, SkillManifest] = {}
        self._aliases: dict[str, str] = {}
        self._lock = RLock()

    def register(self, manifest: SkillManifest) -> None:
        with self._lock:
            if manifest.ref in self._items:
                raise ManifestError(f"Duplicate Skill version: {manifest.ref}")
            self._items[manifest.ref] = manifest
            self._aliases[f"{manifest.id}@{manifest.version.split('.')[0]}"] = manifest.ref
            self._aliases[manifest.id] = manifest.ref

    def resolve(self, ref: str) -> SkillManifest:
        exact = self._aliases.get(ref, ref)
        if exact not in self._items:
            raise ManifestError(f"Unknown Skill: {ref}")
        return self._items[exact]

    def discover(self, root: str | Path) -> "SkillRegistry":
        for path in sorted(Path(root).glob("*/skill.yaml")):
            self.register(SkillManifest.from_yaml(path))
        self.validate_dependencies()
        return self

    def validate_dependencies(self) -> None:
        for manifest in self._items.values():
            for dependency in manifest.dependencies:
                self.resolve(dependency)

    def list(self, *, capability: str | None = None, mcp: bool | None = None) -> list[SkillManifest]:
        items = list(self._items.values())
        if capability:
            items = [item for item in items if capability in item.required_capabilities]
        if mcp is not None:
            items = [item for item in items if item.expose.mcp is mcp]
        return sorted(items, key=lambda item: item.ref)


_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    global _default_registry
    if _default_registry is None:
        root = Path(__file__).resolve().parents[1] / "skills"
        _default_registry = SkillRegistry().discover(root)
    return _default_registry

