import type {
  AskResponse,
  AttachmentResult,
  ConversationTurn,
  ExerciseState,
  Language,
  Readiness,
  SessionIdentity,
  MistakeQuestion,
  LearningReport,
  LearningTask,
} from "./types";
import { createParser } from "eventsource-parser";

declare const __SHOWCASE_MODE__: boolean;
export const SHOWCASE_MODE = typeof __SHOWCASE_MODE__ !== "undefined" && __SHOWCASE_MODE__;

let identity: Promise<string | null> | undefined;

async function browserIdentity(): Promise<string | null> {
  identity ??= fetch("/auth/session", { credentials: "same-origin", signal: AbortSignal.timeout(15000) }).then(async (response) => {
    if (response.status === 404) return null;
    if (!response.ok) throw new Error("identity_unavailable");
    return String((await response.json()).user_id);
  });
  try {
    return await identity;
  } catch (error) {
    identity = undefined;
    throw error;
  }
}

async function identityHeaders(scope: SessionIdentity): Promise<Record<string, string>> {
  const userId = await browserIdentity();
  return { "X-User-ID": userId ?? scope.user_id, "X-Session-ID": scope.session_id };
}

export async function askAgent(
  input: {
    query: string;
    user_id: string;
    session_id: string;
    language: Language;
    conversation_history: ConversationTurn[];
    conversation_summary: string;
    exercise_state: ExerciseState | null;
  },
  signal: AbortSignal,
  onDelta?: (text: string) => void,
): Promise<AskResponse> {
  if (SHOWCASE_MODE) return (await import("./showcase")).askShowcase(input, signal);
  const userId = await browserIdentity();
  const payload = { ...input, user_id: userId ?? input.user_id };
  const response = await fetch("/ask/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
      "X-User-ID": payload.user_id,
      "X-Session-ID": input.session_id,
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || error.error || "request_failed");
  }
  if (!response.body) throw new Error("stream_unavailable");
  let result: AskResponse | undefined;
  let completed = false;
  let streamError = false;
  let answer = "";
  const parser = createParser({ onEvent(event) {
    const data = JSON.parse(event.data);
    if ((event.event === "delta" || event.event === "answer") && data.delta) {
      answer += data.delta;
      onDelta?.(answer);
    }
    if (event.event === "result") result = data as AskResponse;
    if (event.event === "error") streamError = true;
    if (event.event === "done") {
      completed = true;
      if (data.ok === false) streamError = true;
    }
  }});
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      parser.feed(decoder.decode(chunk.value, { stream: true }));
    }
    parser.feed(decoder.decode());
    if (!completed || streamError || !result) throw new Error("stream_incomplete");
    return result;
  } finally {
    reader.releaseLock();
  }
}

export async function parseAttachment(
  file: File,
  language: Language,
  scope: SessionIdentity,
): Promise<AttachmentResult> {
  if (SHOWCASE_MODE) throw new Error("showcase_attachment_unavailable");
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("language", language);
  const response = await fetch("/attachments/parse", {
    method: "POST",
    headers: await identityHeaders(scope),
    body,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "attachment_failed");
  }
  return payload as AttachmentResult;
}

export async function sendFeedback(
  traceId: string,
  correct: boolean,
  scope: SessionIdentity,
): Promise<void> {
  if (SHOWCASE_MODE) return (await import("./showcase")).saveFeedback(traceId, correct, scope);
  const response = await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...await identityHeaders(scope) },
    body: JSON.stringify({ trace_id: traceId, correct, comment: "" }),
  });
  if (!response.ok) {
    throw new Error("feedback_failed");
  }
}

async function learningRequest<T>(
  path: string,
  scope: SessionIdentity,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...await identityHeaders(scope) },
    signal: init.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),
  });
  const payload = await response.json();
  if (!response.ok) {
    const code = response.status === 429 ? "queue_busy" : response.status === 503 ? "service_unavailable" : "learning_failed";
    throw new Error(code);
  }
  return payload as T;
}

export function getMistakes(scope: SessionIdentity, signal?: AbortSignal): Promise<MistakeQuestion[]> {
  if (SHOWCASE_MODE) return import("./showcase").then((demo) => demo.getMistakes(scope, signal));
  return learningRequest("/mistakes?limit=200", scope, { signal });
}

export function getLearningReport(scope: SessionIdentity, signal?: AbortSignal): Promise<LearningReport> {
  if (SHOWCASE_MODE) return import("./showcase").then((demo) => demo.getReport(scope, signal));
  return learningRequest("/reports/learning", scope, { signal });
}

export function createLearningReport(scope: SessionIdentity): Promise<LearningTask> {
  if (SHOWCASE_MODE) return import("./showcase").then((demo) => demo.createReport(scope));
  return learningRequest("/tasks", scope, {
    method: "POST",
    body: JSON.stringify({ kind: "learning_report", payload: {} }),
  });
}

export function getLearningTask(taskId: string, scope: SessionIdentity, signal?: AbortSignal): Promise<LearningTask> {
  if (SHOWCASE_MODE) return import("./showcase").then((demo) => demo.getTask(taskId, scope, signal));
  return learningRequest(`/tasks/${encodeURIComponent(taskId)}`, scope, { signal });
}

export async function getReadiness(): Promise<Readiness> {
  if (SHOWCASE_MODE) return { status: "ready", model: { configured: false, provider: "local", name: "示例解析", force_every_math_turn: false } };
  const [readyResponse, runtimeResponse] = await Promise.all([
    fetch("/ready"),
    fetch("/runtime"),
  ]);
  if (!readyResponse.ok || !runtimeResponse.ok) {
    return {
      status: "degraded",
      model: {
        configured: false,
        provider: "",
        name: "",
        force_every_math_turn: false,
      },
    };
  }
  const ready = await readyResponse.json();
  const runtime = await runtimeResponse.json();
  return {
    status: ready.status,
    model: runtime.model,
    retrieval: runtime.retrieval,
  } as Readiness;
}
