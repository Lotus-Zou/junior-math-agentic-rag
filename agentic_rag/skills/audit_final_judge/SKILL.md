---
name: audit-final-judge
description: Independently make the final structured decision for a self-contained first-error audit after two rejected drafts.
---

# Audit Final Judge

Recompute the numbered solution from the first step. Return the earliest
consequential mathematical error as a typed step number and a concise,
checkable explanation.

- Do not trust the generator's proposed step.
- Ignore harmless wording and consistent intermediate rounding.
- Reject step numbers outside the submitted solution.
- Do not require retrieval evidence for a complete, self-contained audit.
- Use only after both normal Critic passes have rejected a draft.
