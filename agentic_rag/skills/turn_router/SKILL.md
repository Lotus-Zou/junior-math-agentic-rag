---
name: turn_router
description: Route a junior-mathematics conversation turn by its current intent and active exercise state.
---

# math.turn_router

Run before any answer checker or retrieval operation. Distinguish answer submissions,
conceptual follow-ups, hints, solution reveals, exercise generation, difficulty changes,
knowledge questions, new problems, and problem switches.

## Boundaries

- Never treat a question such as "why" or "explain" as a student answer submission.
- Explicit requests for a full solution take precedence over words such as "stuck".
- Preserve the active exercise in `routed_query` for contextual RAG.
- Do not call models, retrievers, FastAPI, MCP, or stateful stores.
