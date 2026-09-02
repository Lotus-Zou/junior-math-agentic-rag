import type {
  AskResponse,
  AttachmentResult,
  ConversationTurn,
  ExerciseState,
  Language,
  Readiness,
} from "./types";

export async function askAgent(
  input: {
    query: string;
    language: Language;
    conversation_history: ConversationTurn[];
    conversation_summary: string;
    exercise_state: ExerciseState | null;
  },
  signal: AbortSignal,
): Promise<AskResponse> {
  const response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "request_failed");
  }
  return payload as AskResponse;
}

export async function parseAttachment(
  file: File,
  language: Language,
): Promise<AttachmentResult> {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("language", language);
  const response = await fetch("/attachments/parse", {
    method: "POST",
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
): Promise<void> {
  const response = await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trace_id: traceId, correct, comment: "" }),
  });
  if (!response.ok) {
    throw new Error("feedback_failed");
  }
}

export async function getReadiness(): Promise<Readiness> {
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
