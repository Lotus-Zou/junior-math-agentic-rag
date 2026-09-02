---
name: answer-final-judge
description: Independently solve a complete self-contained math problem after generation, repair, and two Critic passes fail.
---

# Answer Final Judge

Recompute the problem from its stated conditions and return a typed final
decision plus a checkable corrected answer.

- Run only for complete `problem_solve` and `multiple_choice` turns.
- Do not require retrieval evidence for a complete self-contained problem.
- Do not trust the generator or repair candidate's conclusion.
- Preserve the requested final-answer format.
- Never bypass deterministic equation, division-by-zero, or strict-threshold checks.
- Fall back to targeted clarification when the conditions are genuinely incomplete.
