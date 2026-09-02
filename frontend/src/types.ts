export type Language = "zh" | "en";
export type Theme = "light" | "dark";
export type ResponseType =
  | "verified_answer"
  | "guided_exercise"
  | "clarification_required"
  | "supported_refusal"
  | "general_answer";

export interface ConversationTurn {
  role: "student" | "tutor";
  content: string;
}

export interface ExerciseState {
  exercise_id?: string;
  session_id?: string;
  topic: string;
  grade?: number;
  difficulty?: number;
  difficulty_delta?: number;
  exercise_type?: string;
  template_id?: string;
  fingerprint?: string;
  knowledge_points?: string[];
}

export interface Source {
  chunk_id?: string | null;
  source: string;
  chapter?: string | null;
  rank?: number | null;
}

export interface AgentMetrics {
  tool_calls?: number;
  model_attempts?: number;
  model_successes?: number;
  model_failures?: number;
  latency_ms?: number;
  model_provider?: string;
  model_name?: string;
  llm_required?: boolean;
  execution_path?: string;
}

export interface Readiness {
  status: "ready" | "degraded";
  checks?: Record<string, boolean>;
  model: {
    configured: boolean;
    provider: string;
    name: string;
    force_every_math_turn: boolean;
  };
  retrieval?: {
    chunk_count: number;
    dense_enabled: boolean;
    embedding_model: string;
    graph_nodes: number;
    web_search_configured?: boolean;
  };
}

export interface AskResponse {
  response_type: ResponseType;
  answer: string;
  trace_id: string;
  intent: string;
  knowledge_points: string[];
  sources: Source[];
  validation_passed: boolean;
  conversation_history: ConversationTurn[];
  conversation_summary: string;
  exercise_state: ExerciseState | null;
  metrics: AgentMetrics;
  cached: boolean;
  clarification?: { missing: string[] } | null;
}

export interface AttachmentResult {
  status: string;
  problem_text: string;
  student_answer?: string;
  confidence?: number;
  warnings?: string[];
  trace_id?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  wrongAnswer?: string;
  response?: AskResponse;
  createdAt: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  history: ConversationTurn[];
  summary: string;
  exercise: ExerciseState | null;
  updatedAt: number;
}
