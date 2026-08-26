import unittest
import time

from langchain_core.messages import AIMessage

from agentic_rag.chains import message_text
from agentic_rag.fast_path import build_fast_response
from agentic_rag.guardrails import ensure_tool_allowed, guided_answer_violations
from agentic_rag.knowledge_graph import MathKnowledgeGraph
from agentic_rag.math_retriever import fuse_rankings
from agentic_rag.math_taxonomy import adaptive_chunk_size, classify_math_text, extract_formulas, tokenize_math
from agentic_rag.math_validation import deterministic_equation_answer, deterministic_math_checks
from agentic_rag.nodes import (
    _actionable_missing_conditions,
    _student_attempt_from_query,
    _usable_react_answer,
    classify_knowledge_node,
    parse_question_node,
    rewrite_query_node,
)
from evaluation.evaluation import keyword_recall_at_k, load_and_validate_dataset, quality_gate
from evaluation.generate_dataset import build_rows


class MathPipelineTests(unittest.TestCase):
    def test_geometry_exercise_request_is_fast_and_does_not_retrieve(self):
        for query in ("出一个几何题我做做", "出一个几何体我做做"):
            with self.subTest(query=query):
                result = build_fast_response(query, [], language="zh")
                self.assertIsNotNone(result)
                self.assertEqual(result["intent"], "geometry_exercise")
                self.assertEqual(result["critic_report"]["validation_mode"], "local_template")
                self.assertTrue(result["critic_report"]["exercise_answer_hidden"])
                self.assertEqual(result["metrics"]["tool_calls"], 0)
                self.assertEqual(result["sources"][0]["chapter"], "几何")
                self.assertIn("答案暂不展示", result["answer"])
                self.assertNotIn("知识库没有召回", result["answer"])

    def test_generated_geometry_exercise_keeps_context_for_answer_check(self):
        exercise = build_fast_response("出一个几何体我做做", [], language="zh")
        problem = exercise["answer"]
        self.assertIn("等腰三角形", problem)
        correct = build_fast_response("∠B=70°，∠C=70°", exercise["conversation_history"], language="zh")
        self.assertEqual(correct["intent"], "geometry_answer_check")
        self.assertTrue(correct["critic_report"]["student_answer_correct"])
        self.assertIn("检查通过", correct["answer"])
        hint = build_fast_response("再提示一下", exercise["conversation_history"], language="zh")
        self.assertEqual(hint["intent"], "geometry_hint")
        self.assertIn("答案继续隐藏", hint["answer"])
        self.assertEqual(hint["metrics"]["tool_calls"], 0)

    def test_similar_exercise_fast_path_hides_answer_and_skips_critic(self):
        history = [
            {"role": "student", "content": "解方程 2x+3=11"},
            {"role": "tutor", "content": "分步过程：2x=8，所以 x = 4。"},
        ]
        result = build_fast_response("再出一个类似的题", history, language="zh")
        self.assertIsNotNone(result)
        self.assertEqual(result["intent"], "similar_exercise")
        self.assertEqual(result["critic_report"]["validation_mode"], "local_template")
        self.assertTrue(result["critic_report"]["exercise_answer_hidden"])
        self.assertNotRegex(result["answer"], r"x\s*=\s*-?\d+\s*(?:。|$)")
        self.assertEqual(result["metrics"]["tool_calls"], 0)

    def test_linear_equation_api_fast_path_is_locally_verified(self):
        result = build_fast_response("解方程 2x+3=11", [], language="zh")
        self.assertIsNotNone(result)
        self.assertEqual(result["critic_report"]["validation_mode"], "local_sympy")
        self.assertTrue(result["validation_passed"])
        self.assertIn("x = 4", result["answer"])
        self.assertEqual(result["metrics"]["tool_calls"], 0)

    def test_fast_path_extracts_explicit_student_attempt(self):
        query = "解方程 2x+3=11。\n\n学生错误作答: 2x=11+3"
        self.assertEqual(_student_attempt_from_query(query), "2x=11+3")

    def test_linear_equation_uses_local_planning_fast_path(self):
        now = time.time()
        state = {
            "query": "解方程 2x+3=11。学生错误作答: 2x=11+3",
            "correction_attempts": 0,
            "step_count": 0,
            "deadline_at": now + 300,
            "trace_events": [],
        }
        parsed = parse_question_node(state)
        self.assertTrue(parsed["fast_path"])
        rewritten = rewrite_query_node({**state, **parsed})
        classified = classify_knowledge_node({**state, **parsed, **rewritten})
        self.assertEqual(classified["chapter"], "代数")
        self.assertIn("一元一次方程", classified["knowledge_points"])

    def test_knowledge_query_ignores_optional_clarifications(self):
        missing = ["未指定教材版本、年级及具体章节", "需要用户提供数轴定理依据", "未说明‘它’指代哪个定理"]
        self.assertEqual(_actionable_missing_conditions("knowledge_query", missing), ["未说明‘它’指代哪个定理"])
        self.assertEqual(_actionable_missing_conditions("solve", missing), missing)

    def test_incomplete_react_placeholder_falls_back_to_generation(self):
        self.assertFalse(_usable_react_answer("Sorry, need more steps to process this request."))
        self.assertTrue(_usable_react_answer("知识点定位\n" + "依据教材说明不等式性质。" * 12))

    def test_responses_content_blocks_are_normalized(self):
        message = AIMessage(content=[{"type": "text", "text": "分步解答", "annotations": []}])
        self.assertEqual(message_text(message), "分步解答")
        mixed = AIMessage(content=[{"type": "reasoning", "summary": []}, {"type": "text", "text": "最终答案"}])
        self.assertEqual(message_text(mixed), "最终答案")

    def test_math_classification(self):
        cases = [
            ("解一元二次方程 x^2-5x+6=0", "代数"),
            ("## 代数：不等式\n数轴上用空心圆表示端点", "代数"),
            ("一次函数 y=2x+1 的图像", "函数"),
            ("证明两个三角形全等", "几何"),
            ("求这组数据的平均数与方差", "统计与概率"),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                self.assertEqual(classify_math_text(query).chapter, expected)

    def test_formula_aware_chunk_sizes(self):
        dense = "公式：$x=(-b±\\sqrt{b^2-4ac})/(2a)$，并有 $\\Delta=b^2-4ac$，且 $a!=0$。"
        medium = "一次函数满足 y=kx+b。"
        concept = "相似三角形的对应角相等，对应边成比例。证明前先确定对应关系。"
        self.assertEqual(adaptive_chunk_size(dense), 280)
        self.assertEqual(adaptive_chunk_size(medium), 380)
        self.assertEqual(adaptive_chunk_size(concept), 700)
        self.assertTrue(all(200 <= value <= 800 for value in (280, 380, 700)))
        self.assertGreaterEqual(len(extract_formulas(dense)), 2)

    def test_math_tokenizer_preserves_symbols_and_chinese_terms(self):
        tokens = tokenize_math("一元二次方程 $x^2-5x+6=0$")
        self.assertIn("^", tokens)
        self.assertIn("=", tokens)
        self.assertIn("方程", tokens)

    def test_rrf_rewards_cross_channel_hits(self):
        scores = fuse_rankings(["dense", "both"], ["both", "bm25"], ["graph", "both"])
        self.assertGreater(scores["both"], scores["dense"])
        self.assertGreater(scores["both"], scores["bm25"])
        self.assertGreater(scores["both"], scores["graph"])

    def test_graph_rag_expands_prerequisites(self):
        graph = MathKnowledgeGraph(path="__missing_test_graph__.json")
        expanded = graph.expand(["二次函数"])
        self.assertIn("一元二次方程", expanded)
        self.assertIn("一次函数", expanded)

    def test_deterministic_equation_check(self):
        valid = deterministic_math_checks("解方程 3(x-2)=2x+5", "分步过程 x=11", 0)
        invalid = deterministic_math_checks("解方程 3(x-2)=2x+5", "分步过程 x=10", 0)
        self.assertTrue(valid["passed"])
        self.assertFalse(invalid["passed"])

    def test_equation_check_allows_explaining_a_wrong_candidate(self):
        result = deterministic_math_checks(
            "解方程 2x+3=11",
            "错因分析：错误步骤会得到 x=7。分步过程：2x=8，所以 x=4。自检：2×4+3=11。",
            0,
        )
        self.assertTrue(result["passed"])

    def test_equation_check_accepts_query_rewrite_latex_delimiters(self):
        result = deterministic_math_checks(
            r"解一元一次方程 \(5x+7=32\)，错误作答为 \(5x=32+7\)",
            "知识点定位 [1]\n解题思路\n分步过程：5x=25，所以 x=5。\n自检：5*5+7=32。",
            1,
        )
        self.assertTrue(result["passed"])

    def test_local_equation_fallback_is_complete_and_verifiable(self):
        answer = deterministic_equation_answer(
            "解方程 2x+3=11",
            "我写成 2x=11+3",
            document_count=1,
            language="zh",
        )
        self.assertIn("错因分析", answer)
        self.assertIn("分步过程", answer)
        self.assertIn("x = 4", answer)
        self.assertTrue(deterministic_math_checks("解方程 2x+3=11", answer, 1)["passed"])

    def test_guardrails_and_tool_whitelist(self):
        self.assertTrue(guided_answer_violations("答案：x=2"))
        ensure_tool_allowed("math_retrieval_skill")
        with self.assertRaises(PermissionError):
            ensure_tool_allowed("web_search")

    def test_keyword_recall_at_k(self):
        contexts = ["不等式两边乘以负数时方向改变", "数轴上表示解集"]
        self.assertEqual(keyword_recall_at_k(contexts, "不等式|负数|方向", 1), 1.0)
        self.assertEqual(keyword_recall_at_k(contexts, "不等式|数轴", 1), 0.5)

    def test_benchmark_has_risk_strata(self):
        frame = load_and_validate_dataset()
        self.assertEqual(len(frame), 1000)
        self.assertEqual(set(frame["case_type"]), {"normal", "hallucination_risk", "colloquial"})
        self.assertEqual(len(build_rows()), 1000)

    def test_agent_state_accepts_response_language(self):
        from agentic_rag.state import AgentState

        state: AgentState = {"query": "解方程", "response_language": "zh"}
        self.assertEqual(state["response_language"], "zh")

    def test_responses_model_configuration(self):
        from agentic_rag.chains import generator_llm
        from config import MODEL_REASONING_EFFORT

        if generator_llm.model_name == "gpt-5.6-sol":
            self.assertTrue(generator_llm.use_responses_api)
            self.assertEqual(generator_llm.reasoning, {"effort": MODEL_REASONING_EFFORT})
            self.assertFalse(generator_llm.store)

    def test_quality_gate(self):
        good = {"context_precision": 0.8, "context_recall": 0.8, "faithfulness": 0.8, "answer_relevance": 0.8, "knowledge_point_accuracy": 0.9, "direct_answer_violation_rate": 0.0}
        passed, failures = quality_gate(good)
        self.assertTrue(passed)
        self.assertFalse(failures)
        good["faithfulness"] = 0.2
        self.assertFalse(quality_gate(good)[0])


if __name__ == "__main__":
    unittest.main()
