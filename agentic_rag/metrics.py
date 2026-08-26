# -*- coding: utf-8 -*-
"""Prometheus metrics for the Agent runtime."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("math_agent_requests_total", "Total math Agent requests", ["status"])
LATENCY = Histogram("math_agent_latency_seconds", "End-to-end Agent latency")
TOOL_CALLS = Counter("math_agent_tool_calls_total", "Whitelisted tool calls")
HALLUCINATIONS = Counter("math_agent_hallucinations_detected_total", "Draft hallucinations detected by Critic")
CRITIC_FAILURES = Counter("math_agent_critic_failures_total", "Drafts rejected by independent Critic")
TOKENS = Counter("math_agent_tokens_total", "Reported model tokens", ["kind"])
FEEDBACK = Counter("math_agent_feedback_total", "User feedback labels", ["correct"])


def observe_state(state: dict, latency_seconds: float) -> None:
    REQUESTS.labels("success" if state.get("response") else "failure").inc()
    LATENCY.observe(latency_seconds)
    metrics = state.get("metrics", {})
    TOOL_CALLS.inc(float(metrics.get("tool_calls", 0) or 0))
    HALLUCINATIONS.inc(float(metrics.get("hallucinations_detected", 0) or 0))
    CRITIC_FAILURES.inc(float(metrics.get("critic_failures", 0) or 0))
    usage = metrics.get("generation_tokens", {}) or {}
    for source, target in (("input_tokens", "input"), ("output_tokens", "output"), ("total_tokens", "total"), ("prompt_tokens", "input"), ("completion_tokens", "output")):
        if usage.get(source):
            TOKENS.labels(target).inc(float(usage[source]))
