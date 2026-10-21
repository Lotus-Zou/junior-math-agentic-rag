---
name: answer_repair
description: Repair one rejected RAG answer using its Critic report and unchanged textbook evidence.
---

# math.answer_repair

Run exactly once after the first independent Critic fails. Correct every listed issue without adding unsupported facts, then send the repaired draft to a second Critic.
