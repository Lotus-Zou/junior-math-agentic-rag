import * as Dialog from "@radix-ui/react-dialog";
import {
  Bot,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  ShieldCheck,
  X,
  XCircle,
} from "lucide-react";
import { text } from "../i18n";
import type { AskResponse, Language } from "../types";

interface InspectorProps {
  open: boolean;
  response: AskResponse | null;
  language: Language;
  onOpenChange: (open: boolean) => void;
}

export function Inspector({
  open,
  response,
  language,
  onOpenChange,
}: InspectorProps) {
  const metrics = response?.metrics;
  const showcase = metrics?.execution_path === "showcase";
  const successes = Number(metrics?.model_successes ?? metrics?.tool_calls ?? 0);
  const attempts = Number(metrics?.model_attempts ?? successes);
  const failures = Number(metrics?.model_failures ?? 0);

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="inspector-drawer" aria-describedby={undefined}>
          <header className="inspector-header">
            <div>
              <span>{text(language, "evidence")}</span>
              <Dialog.Title>{text(language, "sources")}</Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <button
                className="icon-button"
                type="button"
                aria-label={text(language, "close")}
              >
                <X size={18} />
              </button>
            </Dialog.Close>
          </header>

          {response && (
            <>
              <section className="inspector-section">
                <h3>
                  <Bot size={16} />
                  {showcase ? (language === "zh" ? "本地示例记录" : "Local example record") : text(language, "agentRun")}
                </h3>
                {showcase ? <p className="source-empty">{language === "zh" ? "预先编写的数学解析。在线模型调用 0 次；未执行实时检索、Critic 或生产性能测量。" : "Curated math explanation. Zero model calls; no live retrieval, Critic, or production measurement."}</p> : <><div className="model-runtime">
                  <span>{metrics?.model_provider || "API"}</span>
                  <strong>{metrics?.model_name || "LLM"}</strong>
                  <small>{metrics?.execution_path || "turn_router"}</small>
                </div>
                <div className="metric-grid">
                  <div>
                    <span>{attempts}</span>
                    <small>Attempts</small>
                  </div>
                  <div>
                    <span>{successes}</span>
                    <small>Success</small>
                  </div>
                  <div className={failures ? "metric-failed" : ""}>
                    <span>{failures}</span>
                    <small>Failed</small>
                  </div>
                </div>
                <div className="run-facts">
                  <p>
                    <Clock3 size={15} />
                    {text(language, "latency")}{" "}
                    {((metrics?.latency_ms || 0) / 1000).toFixed(1)}s
                  </p>
                  <p>
                    {response.validation_passed ? (
                      <CheckCircle2 size={15} />
                    ) : (
                      <XCircle size={15} />
                    )}
                    {response.validation_passed
                      ? text(language, "verified")
                      : text(language, "needsInfo")}
                  </p>
                  <p>
                    <ShieldCheck size={15} />
                    {response.cached ? "Cache" : `Trace ${response.trace_id.slice(0, 8)}`}
                  </p>
                </div>
                </>}
              </section>

              <section className="inspector-section">
                <h3>
                  <Database size={16} />
                  {text(language, "sources")}
                </h3>
                {response.sources.length === 0 ? (
                  <p className="source-empty">{text(language, "noSources")}</p>
                ) : (
                  <div className="source-stack">
                    {response.sources.map((source, index) => (
                      <article
                        className="source-entry"
                        key={source.chunk_id || `${source.source}-${index}`}
                      >
                        <FileText size={17} />
                        <div>
                          <strong>{source.source || `教材片段 ${index + 1}`}</strong>
                          <p>
                            {source.chapter || "初中数学"}
                            <span>#{source.rank || index + 1}</span>
                          </p>
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>

              {response.knowledge_points.length > 0 && (
                <section className="inspector-section">
                  <h3>Knowledge</h3>
                  <div className="knowledge-list">
                    {response.knowledge_points.map((point) => (
                      <span key={point}>{point}</span>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
