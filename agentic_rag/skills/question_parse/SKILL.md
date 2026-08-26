---
name: question_parse
description: Use this repository Skill for question parse in the junior-mathematics tutoring pipeline. It applies whenever this capability is executed by Workflow, Tool-Calling, MCP, or evaluation.
---

# math.question_parse

Validate input against `QueryInput`, execute only the declared question parse capability, and return `QuestionParseOutput`.

## Boundaries

- Stay within the junior-high mathematics curriculum.
- Do not invent missing conditions, evidence, citations, or student history.
- Do not call FastAPI, LangGraph, LangChain Tool, MCP, tracing, cache, or databases directly.
- Return structured output; let the Runtime apply policy, timeout, retry, telemetry, and safe errors.

## Evaluation

Run the normal and invalid-input cases in `tests.yaml`. Changes to output meaning require a new semantic version and differential pipeline tests.
