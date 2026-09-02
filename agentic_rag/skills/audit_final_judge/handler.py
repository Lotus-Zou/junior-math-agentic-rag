from agentic_rag.domain.schemas import AnswerEnvelope, AnswerRepairInput
from agentic_rag.skill_handlers import audit_final_judge


def handle(data: AnswerRepairInput, context) -> AnswerEnvelope:
    return audit_final_judge(data, context)
