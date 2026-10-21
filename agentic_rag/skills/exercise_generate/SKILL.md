---
name: exercise_generate
description: Generate an adaptive junior-mathematics exercise after the Turn Router selects a new-exercise or difficulty-adjustment turn.
---

# math.exercise_generate

Start from a deterministically verified seed, let the author Agent vary presentation, and require both parameter validation and an independent Critic before storing private solution state.

## Boundaries

- Never run before `math.turn_router`.
- Never expose solution, answer signature, parameters, or private mastery state.
- Fall back to the verified seed when either model call fails or changes the mathematics.
- Keep generation bounded to one author call and one Critic call.
