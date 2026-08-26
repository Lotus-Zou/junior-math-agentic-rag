"""Machine-readable Skill manifest loading and validation."""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentic_rag.skill_runtime.errors import ManifestError

SKILL_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class Exposure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    langchain: bool = False
    mcp: bool = False


class SkillManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    version: str
    description: str = Field(min_length=10)
    input_schema: str
    output_schema: str
    handler: str
    timeout_ms: int = Field(default=1500, ge=1, le=8000)
    max_attempts: int = Field(default=1, ge=1, le=3)
    idempotent: bool = True
    side_effects: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    fallback: str | None = None
    policies: list[str] = Field(default_factory=list)
    evaluator: str = ""
    expose: Exposure = Field(default_factory=Exposure)

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not SKILL_ID.fullmatch(value):
            raise ValueError("skill id must be a dotted lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        if not SEMVER.fullmatch(value):
            raise ValueError("version must use semantic x.y.z form")
        return value

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @staticmethod
    def import_symbol(path: str) -> Any:
        module_name, separator, name = path.rpartition(".")
        if not separator:
            raise ManifestError(f"Invalid import path: {path}")
        try:
            return getattr(importlib.import_module(module_name), name)
        except (ImportError, AttributeError) as exc:
            raise ManifestError(f"Cannot import {path}: {exc}") from exc

    @property
    def input_model(self):
        return self.import_symbol(self.input_schema)

    @property
    def output_model(self):
        return self.import_symbol(self.output_schema)

    @property
    def handler_callable(self):
        return self.import_symbol(self.handler)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkillManifest":
        source = Path(path)
        try:
            payload = yaml.safe_load(source.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except Exception as exc:
            raise ManifestError(f"Invalid Skill manifest {source}: {exc}") from exc

