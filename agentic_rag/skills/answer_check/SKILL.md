---
name: answer_check
description: Use this repository Skill for answer check in the junior-mathematics tutoring pipeline. It applies whenever this capability is executed by Workflow, Tool-Calling, MCP, or evaluation.
---

# math.answer_check

Validate input against `AnswerCheckInput`, execute only the declared answer check capability, and return `CriticOutput`.

## Boundaries

- Stay within the junior-high mathematics curriculum.
- Do not invent missing conditions, evidence, citations, or student history.
- Do not call FastAPI, LangGraph, LangChain Tool, MCP, tracing, cache, or databases directly.
- Return structured output; let the Runtime apply policy, timeout, retry, telemetry, and safe errors.

## Evaluation

Run the normal and invalid-input cases in `tests.yaml`. Changes to output meaning require a new semantic version and differential pipeline tests.
