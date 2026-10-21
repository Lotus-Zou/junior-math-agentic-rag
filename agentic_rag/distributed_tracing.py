"""W3C trace propagation with bounded, local OpenTelemetry span export."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Lock

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from config import TRACE_PATH


class JsonSpanExporter(SpanExporter):
    def __init__(self):
        self.path = Path(TRACE_PATH) / f"spans-{os.getpid()}.jsonl"
        self.lock = Lock()

    def export(self, spans):
        records = []
        for span in spans:
            records.append(json.dumps({
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
                "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                "name": span.name, "start_time_ns": span.start_time, "end_time_ns": span.end_time,
                "status": span.status.status_code.name, "attributes": dict(span.attributes or {}),
            }, ensure_ascii=False))
        try:
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write("\n".join(records) + "\n")
        except OSError:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS


provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "math-agent")}))
provider.add_span_processor(SimpleSpanProcessor(JsonSpanExporter()))
tracer = provider.get_tracer("math-agent")
propagator = TraceContextTextMapPropagator()


@contextmanager
def span(name: str, carrier: dict[str, str] | None = None, **attributes):
    context = propagator.extract(carrier) if carrier is not None else None
    with tracer.start_as_current_span(name, context=context, attributes=attributes) as current:
        yield current


def trace_headers() -> dict[str, str]:
    carrier: dict[str, str] = {}
    propagator.inject(carrier)
    return carrier
