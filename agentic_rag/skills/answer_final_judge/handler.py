from agentic_rag.domain.schemas import AnswerEnvelope, AnswerRepairInput
from agentic_rag.skill_handlers import answer_final_judge


def handle(data: AnswerRepairInput, context) -> AnswerEnvelope:
    return answer_final_judge(data, context)
