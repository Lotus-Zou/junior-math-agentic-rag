import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as Tooltip from "@radix-ui/react-tooltip";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Bot,
  Check,
  CheckCircle2,
  ChevronDown,
  Clipboard,
  Copy,
  Database,
  GraduationCap,
  Languages,
  Menu,
  Moon,
  RefreshCw,
  Sparkles,
  Sun,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { askAgent, getReadiness, sendFeedback } from "./api";
import { Composer } from "./components/Composer";
import { MathVisual } from "./components/MathVisual";
import { Sidebar } from "./components/Sidebar";
import { text } from "./i18n";
import { newMessage, useChatStore } from "./store";
import type {
  AskResponse,
  ChatMessage,
  ExerciseState,
  Language,
  ResponseType,
} from "./types";

type Grade = "auto" | "7" | "8" | "9";

const Inspector = lazy(() =>
  import("./components/Inspector").then((module) => ({
    default: module.Inspector,
  })),
);
const MarkdownAnswer = lazy(() =>
  import("./components/MarkdownAnswer").then((module) => ({
    default: module.MarkdownAnswer,
  })),
);

function responseLabel(language: Language, type: ResponseType): string {
  const labels: Record<ResponseType, string> = {
    verified_answer: text(language, "verified"),
    guided_exercise: text(language, "practice"),
    clarification_required: text(language, "needsInfo"),
    supported_refusal: text(language, "refused"),
    general_answer: language === "zh" ? "通用回答" : "General answer",
  };
  return labels[type];
}

function exerciseTopic(exercise: ExerciseState, language: Language): string {
  const topics: Record<string, [string, string]> = {
    geometry: ["几何", "Geometry"],
    algebra: ["代数", "Algebra"],
    linear_function: ["一次函数", "Linear function"],
  };
  const pair = topics[exercise.topic] || ["初中数学", "Junior math"];
  return pair[language === "zh" ? 0 : 1];
}

function executionLabel(language: Language, path?: string): string {
  const labels: Record<string, [string, string]> = {
    agentic_rag: ["Agentic RAG", "Agentic RAG"],
    tutor_agent: ["Tutor Agent", "Tutor Agent"],
    exercise_agent: ["出题 Agent", "Exercise Agent"],
    verified_fallback: ["数学核验兜底", "Verified fallback"],
    policy_guard: ["安全护栏", "Policy guard"],
    turn_router: ["Turn Router", "Turn Router"],
    utility_tool: ["时间工具", "Time tool"],
    general_agent: ["通用 Agent", "General Agent"],
  };
  const pair = labels[path || "turn_router"] || labels.turn_router;
  return pair[language === "zh" ? 0 : 1];
}

function AgentRail({
  response,
  language,
}: {
  response: AskResponse;
  language: Language;
}) {
  const modelSuccess = Number(
    response.metrics.model_successes ?? response.metrics.tool_calls ?? 0,
  );
  const steps = [
    { label: text(language, "router"), active: true },
    { label: text(language, "retrieve"), active: response.sources.length > 0 },
    { label: text(language, "generate"), active: modelSuccess > 0 },
    { label: text(language, "critic"), active: response.validation_passed },
  ];
  return (
    <div className="proof-rail" aria-label={text(language, "agentRun")}>
      {steps.map((step) => (
        <div className={step.active ? "proof-node done" : "proof-node"} key={step.label}>
          <span>{step.active && <Check size={10} />}</span>
          <small>{step.label}</small>
        </div>
      ))}
    </div>
  );
}

function MessageActions({
  message,
  language,
  onInspect,
  onRetry,
}: {
  message: ChatMessage;
  language: Language;
  onInspect: (response: AskResponse) => void;
  onRetry: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<"up" | "down" | null>(null);
  const response = message.response;
  if (!response) return null;

  const feedback = async (correct: boolean) => {
    setVote(correct ? "up" : "down");
    try {
      await sendFeedback(response.trace_id, correct);
    } catch {
      setVote(null);
    }
  };

  return (
    <div className="message-actions">
      <button
        type="button"
        onClick={async () => {
          await navigator.clipboard.writeText(message.content);
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1600);
        }}
        aria-label={text(language, "copy")}
      >
        {copied ? <Check size={15} /> : <Copy size={15} />}
        <span>{copied ? text(language, "copied") : text(language, "copy")}</span>
      </button>
      <button type="button" onClick={onRetry} aria-label={text(language, "retry")}>
        <RefreshCw size={15} />
        <span>{text(language, "retry")}</span>
      </button>
      <button
        type="button"
        onClick={() => onInspect(response)}
        aria-label={text(language, "sources")}
      >
        <Database size={15} />
        <span>{text(language, "sources")}</span>
      </button>
      <button
        type="button"
        className={vote === "up" ? "selected" : ""}
        onClick={() => feedback(true)}
        aria-label="Helpful"
      >
        <ThumbsUp size={15} />
      </button>
      <button
        type="button"
        className={vote === "down" ? "selected" : ""}
        onClick={() => feedback(false)}
        aria-label="Report"
      >
        <ThumbsDown size={15} />
      </button>
    </div>
  );
}

function AssistantMessage({
  message,
  query,
  language,
  onInspect,
  onRetry,
}: {
  message: ChatMessage;
  query: string;
  language: Language;
  onInspect: (response: AskResponse) => void;
  onRetry: () => void;
}) {
  const response = message.response;
  const successes = Number(
    response?.metrics.model_successes ?? response?.metrics.tool_calls ?? 0,
  );
  const failures = Number(response?.metrics.model_failures ?? 0);
  return (
    <motion.article
      className="message assistant"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="assistant-mark" aria-hidden="true">
        <Sparkles size={17} />
      </div>
      <div className="assistant-body">
        {response && <AgentRail response={response} language={language} />}
        <MathVisual query={query} answer={message.content} />
        <Suspense fallback={<div className="answer-loading" aria-hidden="true" />}>
          <MarkdownAnswer content={message.content} />
        </Suspense>
        {response && (
          <div className="answer-meta">
            <span
              className={
                response.validation_passed ? "meta-chip valid" : "meta-chip"
              }
            >
              {response.validation_passed && <CheckCircle2 size={13} />}
              {responseLabel(language, response.response_type)}
            </span>
            {successes > 0 && (
              <span className="meta-chip agent">
                <Bot size={13} />
                {response.metrics.model_name || "LLM"} · API {successes}/
                {Number(response.metrics.model_attempts || successes)} · {successes}{" "}
                {text(language, "modelCalls")}
              </span>
            )}
            {failures > 0 && successes === 0 && (
              <span className="meta-chip fallback">
                {text(language, "modelFallback")}
              </span>
            )}
            {response.metrics.latency_ms != null && (
              <span className="meta-chip">
                {text(language, "latency")}{" "}
                {(response.metrics.latency_ms / 1000).toFixed(1)}s
              </span>
            )}
            <span className="meta-chip path">
              {executionLabel(language, response.metrics.execution_path)}
            </span>
            {response.knowledge_points.slice(0, 3).map((point) => (
              <span className="meta-chip" key={point}>
                {point}
              </span>
            ))}
          </div>
        )}
        <MessageActions
          message={message}
          language={language}
          onInspect={onInspect}
          onRetry={onRetry}
        />
      </div>
    </motion.article>
  );
}

function ThinkingMessage({
  language,
  modelName,
}: {
  language: Language;
  modelName?: string;
}) {
  const labels = [
    text(language, "router"),
    text(language, "retrieve"),
    text(language, "generate"),
    text(language, "critic"),
  ];
  return (
    <article className="message assistant thinking-message">
      <div className="assistant-mark thinking">
        <Bot size={17} />
      </div>
      <div className="assistant-body">
        <div className="live-model-call">
          <i />
          <strong>{modelName || "LLM"}</strong>
          <span>{text(language, "callingModel")}</span>
        </div>
        <div className="thinking-track">
          {labels.map((label, index) => (
            <motion.span
              key={label}
              animate={{ opacity: [0.35, 1, 0.35] }}
              transition={{
                duration: 2.4,
                repeat: Infinity,
                delay: index * 0.45,
              }}
            >
              {label}
            </motion.span>
          ))}
        </div>
      </div>
    </article>
  );
}

export default function App() {
  const {
    sessions,
    activeId,
    language,
    theme,
    addMessage,
    replaceLastAssistant,
    applyResponse,
    setLanguage,
    setTheme,
  } = useChatStore();
  const active = sessions.find((session) => session.id === activeId) || sessions[0];
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspected, setInspected] = useState<AskResponse | null>(null);
  const [grade, setGrade] = useState<Grade>("auto");
  const scrollRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const ready = useQuery({
    queryKey: ["readiness"],
    queryFn: getReadiness,
    refetchInterval: 30000,
  });
  const askMutation = useMutation({
    mutationFn: ({
      payload,
      signal,
    }: {
      payload: Parameters<typeof askAgent>[0];
      signal: AbortSignal;
    }) => askAgent(payload, signal),
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title =
      language === "zh"
        ? "数问 · 初中数学 Agent"
        : "MathTrace · Junior Math Agent";
  }, [language, theme]);

  useEffect(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    });
  }, [active?.messages.length, askMutation.isPending]);

  const gradeLabel =
    grade === "auto"
      ? language === "zh"
        ? "自动判断年级"
        : "Auto grade"
      : language === "zh"
        ? `${grade} 年级`
        : `Grade ${grade}`;

  const buildQuery = (
    question: string,
    wrongAnswer: string,
  ): string => {
    let query = question;
    if (grade !== "auto") {
      query = language === "zh" ? `[初中${grade}年级] ${query}` : `[Grade ${grade}] ${query}`;
    }
    if (wrongAnswer) {
      query +=
        language === "zh"
          ? `\n\n学生错误作答：${wrongAnswer}`
          : `\n\nStudent's incorrect attempt: ${wrongAnswer}`;
    }
    return query;
  };

  const runAgent = async (
    question: string,
    wrongAnswer: string,
    replace = false,
  ): Promise<boolean> => {
    if (!active || askMutation.isPending) return false;
    if (!replace) {
      addMessage(newMessage("user", question, { wrongAnswer }));
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    timeoutRef.current = window.setTimeout(() => controller.abort(), 185000);
    try {
      const response = await askMutation.mutateAsync({
        payload: {
          query: buildQuery(question, wrongAnswer),
          language,
          conversation_history: active.history,
          conversation_summary: active.summary,
          exercise_state: active.exercise,
        },
        signal: controller.signal,
      });
      const message = newMessage("assistant", response.answer, { response });
      if (replace) replaceLastAssistant(message);
      else addMessage(message);
      applyResponse(response);
      setInspected(response);
      return true;
    } catch (error) {
      const aborted =
        error instanceof DOMException && error.name === "AbortError";
      if (!replace) {
        addMessage(
          newMessage(
            "assistant",
            aborted ? text(language, "stopped") : text(language, "failed"),
          ),
        );
      }
      return false;
    } finally {
      if (timeoutRef.current) window.clearTimeout(timeoutRef.current);
      controllerRef.current = null;
    }
  };

  const retryLast = async (assistantId: string) => {
    if (!active) return;
    const assistantIndex = active.messages.findIndex(
      (message) => message.id === assistantId,
    );
    const previous = active.messages
      .slice(0, assistantIndex)
      .findLast((message) => message.role === "user");
    if (previous) {
      await runAgent(
        previous.content,
        previous.wrongAnswer || "",
        true,
      );
    }
  };

  const currentExercise = active?.exercise;
  const difficulty = Math.max(
    1,
    Math.min(5, Number(currentExercise?.difficulty || 1)),
  );
  const lastResponse = useMemo(
    () =>
      active?.messages
        .filter((message) => message.response)
        .at(-1)?.response || null,
    [active?.messages],
  );

  return (
    <Tooltip.Provider delayDuration={300}>
      <div className="app-frame">
        <Sidebar
          open={sidebarOpen}
          busy={askMutation.isPending}
          onClose={() => setSidebarOpen(false)}
        />
        <main className="chat-workspace">
          <header className="workspace-bar">
            <div className="workspace-left">
              <button
                id="menuButton"
                className="icon-button mobile-menu"
                type="button"
                onClick={() => setSidebarOpen(true)}
                aria-label={text(language, "menu")}
              >
                <Menu size={20} />
              </button>
              <div className="workspace-title">
                <strong>{active?.title || text(language, "newChat")}</strong>
                <span
                  id="serviceStatus"
                  className={
                    ready.data?.status === "ready"
                      ? "online-state"
                      : "online-state waiting"
                  }
                >
                  <i />
                  {ready.data?.status === "ready" && ready.data.model.configured
                    ? `${ready.data.model.name} · ${text(language, "apiConnected")} · RAG ${ready.data.retrieval?.chunk_count || 0}`
                    : text(language, "degraded")}
                </span>
              </div>
            </div>
            <div className="workspace-actions">
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <button className="select-button" type="button">
                    <GraduationCap size={16} />
                    <span>{gradeLabel}</span>
                    <ChevronDown size={14} />
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content className="context-menu" sideOffset={7}>
                    {(["auto", "7", "8", "9"] as Grade[]).map((item) => (
                      <DropdownMenu.Item
                        className="context-menu-item"
                        onSelect={() => setGrade(item)}
                        key={item}
                      >
                        {item === grade && <Check size={14} />}
                        {item === "auto"
                          ? language === "zh"
                            ? "自动判断年级"
                            : "Auto grade"
                          : language === "zh"
                            ? `${item} 年级`
                            : `Grade ${item}`}
                      </DropdownMenu.Item>
                    ))}
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
              <button
                id="languageButton"
                data-language={language === "zh" ? "en" : "zh"}
                className="icon-button"
                type="button"
                onClick={() => setLanguage(language === "zh" ? "en" : "zh")}
                aria-label={text(language, "language")}
              >
                <Languages size={18} />
              </button>
              <button
                id="themeButton"
                className="icon-button"
                type="button"
                onClick={() => setTheme(theme === "light" ? "dark" : "light")}
                aria-label={text(language, "theme")}
              >
                {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
              </button>
              <button
                id="evidenceButton"
                className="evidence-button"
                type="button"
                disabled={!lastResponse}
                onClick={() => {
                  setInspected(lastResponse);
                  setInspectorOpen(true);
                }}
              >
                <Clipboard size={16} />
                <span>{text(language, "evidence")}</span>
              </button>
            </div>
          </header>

          {currentExercise && (
            <section className="exercise-bar" id="exerciseContext">
              <div className="exercise-name">
                <span>{text(language, "currentExercise")}</span>
                <strong>{exerciseTopic(currentExercise, language)}</strong>
                {currentExercise.grade && (
                  <small>
                    {language === "zh"
                      ? `${currentExercise.grade} 年级`
                      : `Grade ${currentExercise.grade}`}
                  </small>
                )}
                {currentExercise.knowledge_points?.[0] && (
                  <small>{currentExercise.knowledge_points[0]}</small>
                )}
              </div>
              <div className="difficulty">
                <button
                  type="button"
                  disabled={askMutation.isPending || difficulty <= 1}
                  onClick={() =>
                    runAgent(language === "zh" ? "太难了" : "too hard", "")
                  }
                >
                  {text(language, "easier")}
                </button>
                <div className="difficulty-dots" aria-hidden="true">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <i className={level <= difficulty ? "active" : ""} key={level} />
                  ))}
                </div>
                <span id="difficultyLabel">
                  {text(language, "difficulty")} {difficulty} / 5
                </span>
                <button
                  id="harderButton"
                  type="button"
                  disabled={askMutation.isPending || difficulty >= 5}
                  onClick={() =>
                    runAgent(language === "zh" ? "太简单了" : "too easy", "")
                  }
                >
                  {text(language, "harder")}
                </button>
              </div>
            </section>
          )}

          <div className="conversation-scroll" ref={scrollRef}>
            <div className="conversation-column">
              {active?.messages.length === 0 && (
                <section className="welcome-state" id="welcome">
                  <div className="welcome-glyph" aria-hidden="true">
                    <span>∵</span>
                    <span>∴</span>
                  </div>
                  <h1>{text(language, "welcome")}</h1>
                </section>
              )}

              <AnimatePresence initial={false}>
                {active?.messages.map((message, index) =>
                  message.role === "user" ? (
                    <motion.article
                      className="message user"
                      key={message.id}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <div className="user-bubble">
                        <p>{message.content}</p>
                        {message.wrongAnswer && (
                          <div className="wrong-submission">
                            <span>{text(language, "wrongAnswer")}</span>
                            {message.wrongAnswer}
                          </div>
                        )}
                      </div>
                    </motion.article>
                  ) : (
                    <AssistantMessage
                      key={message.id}
                      message={message}
                      query={
                        active.messages
                          .slice(0, index)
                          .findLast((item) => item.role === "user")?.content || ""
                      }
                      language={language}
                      onInspect={(response) => {
                        setInspected(response);
                        setInspectorOpen(true);
                      }}
                      onRetry={() => retryLast(message.id)}
                    />
                  ),
                )}
              </AnimatePresence>
              {askMutation.isPending && (
                <ThinkingMessage
                  language={language}
                  modelName={ready.data?.model.name}
                />
              )}
            </div>
          </div>

          <Composer
            language={language}
            busy={askMutation.isPending}
            onSend={runAgent}
            onStop={() => controllerRef.current?.abort()}
          />
        </main>

        {inspectorOpen && (
          <Suspense fallback={null}>
            <Inspector
              open={inspectorOpen}
              response={inspected}
              language={language}
              onOpenChange={setInspectorOpen}
            />
          </Suspense>
        )}
      </div>
    </Tooltip.Provider>
  );
}
