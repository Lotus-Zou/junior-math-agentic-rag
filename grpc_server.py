"""Native gRPC service for the gateway's internal request path."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from functools import wraps

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from fastapi import HTTPException

import app
from proto import math_agent_pb2, math_agent_pb2_grpc
from agentic_rag.task_queue import enqueue
from agentic_rag.streaming import register as register_stream, unregister as unregister_stream
from agentic_rag.distributed_tracing import span


def traced_rpc(function):
    @wraps(function)
    async def wrapped(self, request, context):
        carrier = dict(context.invocation_metadata())
        with span("grpc." + function.__name__, carrier=carrier):
            return await function(self, request, context)
    return wrapped


def traced_stream(function):
    @wraps(function)
    async def wrapped(self, request, context):
        carrier = dict(context.invocation_metadata())
        with span("grpc." + function.__name__, carrier=carrier):
            async for event in function(self, request, context):
                yield event
    return wrapped


class MathAgentService(math_agent_pb2_grpc.MathAgentServicer):
    @staticmethod
    def _metadata(context, name: str) -> str | None:
        name = name.lower()
        for key, value in context.invocation_metadata():
            if key.lower() == name:
                return value
        return None

    async def _identity(self, context, user_id: str, session_id: str) -> tuple[str, str]:
        header_user_id = self._metadata(context, "x-user-id") or user_id
        header_session_id = self._metadata(context, "x-session-id") or session_id
        authorization = self._metadata(context, "authorization")
        signature = self._metadata(context, "x-user-signature")
        try:
            return app._resolve_identity(
                app.AskRequest(
                    query="grpc",
                    user_id=header_user_id or "anonymous",
                    session_id=header_session_id or "anonymous",
                ),
                header_user_id,
                header_session_id,
                authorization,
                signature,
            )
        except HTTPException as exc:
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED
                if exc.status_code in {401, 403}
                else grpc.StatusCode.INVALID_ARGUMENT,
                str(exc.detail),
            )
            raise AssertionError("context.abort did not terminate the RPC")

    @traced_rpc
    async def Ask(self, request, context):
        try:
            payload = json.loads(request.payload_json) if request.payload_json else {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "query": request.query,
                "language": request.language or "zh",
                "conversation_history": [
                    {"role": item.role, "content": item.content}
                    for item in request.history
                ],
            }
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be valid JSON")
            raise AssertionError("context.abort did not terminate the RPC")
        if not isinstance(payload, dict):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be an object")
            raise AssertionError("context.abort did not terminate the RPC")

        user_id, session_id = await self._identity(
            context,
            str(payload.get("user_id") or request.user_id or "anonymous"),
            str(payload.get("session_id") or request.session_id or "anonymous"),
        )
        payload["user_id"] = user_id
        payload["session_id"] = session_id
        try:
            model_request = app.AskRequest.model_validate(payload)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid ask request: {exc}")
            raise AssertionError("context.abort did not terminate the RPC")
        try:
            result = await app.ask(
                model_request,
                x_user_id=user_id,
                x_session_id=session_id,
                authorization=self._metadata(context, "authorization"),
                x_user_signature=self._metadata(context, "x-user-signature"),
            )
        except HTTPException as exc:
            code = {
                400: grpc.StatusCode.INVALID_ARGUMENT,
                401: grpc.StatusCode.UNAUTHENTICATED,
                403: grpc.StatusCode.PERMISSION_DENIED,
                429: grpc.StatusCode.RESOURCE_EXHAUSTED,
            }.get(exc.status_code, grpc.StatusCode.INTERNAL)
            await context.abort(code, str(exc.detail))
            raise AssertionError("context.abort did not terminate the RPC")
        return math_agent_pb2.AskResponse(
            trace_id=result.get("trace_id", ""),
            answer=result.get("answer", ""),
            intent=result.get("intent", ""),
            validated=bool(result.get("validation_passed")),
            sources=[math_agent_pb2.Source(chunk_id=str(item.get("chunk_id") or ""), source=item.get("source", ""), chapter=str(item.get("chapter") or "")) for item in result.get("sources", [])],
            payload_json=json.dumps(result, ensure_ascii=False),
        )

    @traced_stream
    async def AskStream(self, request, context):
        try:
            payload = json.loads(request.payload_json) if request.payload_json else {
                "user_id": request.user_id,
                "session_id": request.session_id,
                "query": request.query,
                "language": request.language or "zh",
                "conversation_history": [{"role": item.role, "content": item.content} for item in request.history],
            }
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be valid JSON")
            return
        if not isinstance(payload, dict):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be an object")
        user_id, session_id = await self._identity(context, str(payload.get("user_id") or request.user_id or "anonymous"), str(payload.get("session_id") or request.session_id or "anonymous"))
        payload["user_id"] = user_id
        payload["session_id"] = session_id
        payload["request_id"] = str(payload.get("request_id") or request.request_id or uuid.uuid4())
        payload["stream_output"] = True
        try:
            model_request = app.AskRequest.model_validate(payload)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid ask request: {exc}")
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[dict] = asyncio.Queue(maxsize=256)

        def publish(item: dict) -> None:
            def offer() -> None:
                try:
                    events.put_nowait(item)
                except asyncio.QueueFull:
                    return
            try:
                loop.call_soon_threadsafe(offer)
            except RuntimeError:
                return

        register_stream(model_request.request_id, publish)
        task = asyncio.create_task(app.ask(model_request, x_user_id=user_id, x_session_id=session_id, authorization=self._metadata(context, "authorization"), x_user_signature=self._metadata(context, "x-user-signature")))
        try:
            yield math_agent_pb2.AskEvent(event="ready", payload_json=json.dumps({"status": "accepted", "trace_id": model_request.request_id}, ensure_ascii=False))
            while not task.done() or not events.empty():
                try:
                    item = await asyncio.wait_for(events.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                event_name = item.get("event", "progress")
                yield math_agent_pb2.AskEvent(event=event_name, delta=str(item.get("delta", "")), payload_json=json.dumps({key: value for key, value in item.items() if key != "event"}, ensure_ascii=False))
            try:
                result = await task
            except HTTPException as exc:
                code = {
                    400: grpc.StatusCode.INVALID_ARGUMENT,
                    401: grpc.StatusCode.UNAUTHENTICATED,
                    403: grpc.StatusCode.PERMISSION_DENIED,
                    429: grpc.StatusCode.RESOURCE_EXHAUSTED,
                    503: grpc.StatusCode.UNAVAILABLE,
                    504: grpc.StatusCode.DEADLINE_EXCEEDED,
                }.get(exc.status_code, grpc.StatusCode.INTERNAL)
                await context.abort(code, "ask request rejected")
                return
            yield math_agent_pb2.AskEvent(event="result", payload_json=json.dumps(result, ensure_ascii=False))
            yield math_agent_pb2.AskEvent(event="done", payload_json="{}")
        finally:
            unregister_stream(model_request.request_id)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def EnqueueLearningTask(self, request, context):
        user_id, session_id = await self._identity(context, request.user_id, request.session_id)
        try:
            payload = json.loads(request.payload_json or "{}")
        except json.JSONDecodeError:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "payload_json must be valid JSON")
            raise AssertionError("context.abort did not terminate the RPC")
        task_id = enqueue(request.kind, payload, user_id=user_id, session_id=session_id)
        return math_agent_pb2.TaskAccepted(task_id=task_id, status="queued")

    async def Readiness(self, request, context):
        checks = app._readiness_checks()
        return math_agent_pb2.ReadinessResponse(
            status="ready" if all(checks.values()) else "degraded",
            checks_json=json.dumps(checks, ensure_ascii=False, sort_keys=True),
        )


async def serve() -> None:
    app.memory.initialize_memory_db()
    app.domain_store.initialize_domain_schema()
    server = grpc.aio.server()
    health_service = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    math_agent_pb2_grpc.add_MathAgentServicer_to_server(MathAgentService(), server)
    server.add_insecure_port(f"[::]:{os.getenv('GRPC_PORT', '50051')}")
    await server.start()

    async def update_health():
        while True:
            checks = await asyncio.to_thread(app._readiness_checks)
            await health_service.set("", health_pb2.HealthCheckResponse.SERVING)
            await health_service.set("readiness", health_pb2.HealthCheckResponse.SERVING if all(checks.values()) else health_pb2.HealthCheckResponse.NOT_SERVING)
            await asyncio.sleep(5)

    monitor = asyncio.create_task(update_health())
    try:
        await server.wait_for_termination()
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(serve())
