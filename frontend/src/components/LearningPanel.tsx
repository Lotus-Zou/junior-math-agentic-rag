import { useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { useMutation, useQuery } from "@tanstack/react-query";
import { BookOpen, ChartNoAxesCombined, Download, LoaderCircle, RefreshCw, X } from "lucide-react";
import { createLearningReport, getLearningReport, getLearningTask, getMistakes } from "../api";
import type { Language, LearningReport, SessionIdentity } from "../types";
import { MarkdownAnswer } from "./MarkdownAnswer";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  scope: SessionIdentity;
  language: Language;
  busy: boolean;
  onReview: (question: string) => void;
}

function downloadReport(report: LearningReport) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "learning-report.json";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function errorLabel(value: string | null, zh: boolean): string {
  if (!value) return zh ? "未分类" : "Unclassified";
  if (value === "error_analysis") return zh ? "错题分析" : "Mistake analysis";
  return value;
}

export function LearningPanel({ open, onOpenChange, scope, language, busy, onReview }: Props) {
  const zh = language === "zh";
  const [tab, setTab] = useState<"mistakes" | "report">("mistakes");
  const [taskId, setTaskId] = useState<string | null>(null);
  const mistakes = useQuery({
    queryKey: ["mistakes", scope.user_id, scope.session_id],
    queryFn: ({ signal }) => getMistakes(scope, signal),
    enabled: open && tab === "mistakes",
    retry: false,
  });
  const report = useQuery({
    queryKey: ["learning-report", scope.user_id, scope.session_id],
    queryFn: ({ signal }) => getLearningReport(scope, signal),
    enabled: open && tab === "report",
    retry: false,
  });
  const task = useQuery({
    queryKey: ["learning-task", scope.user_id, scope.session_id, taskId],
    queryFn: ({ signal }) => getLearningTask(taskId!, scope, signal),
    enabled: open && !!taskId,
    retry: false,
    refetchInterval: (query) => ["done", "failed"].includes(query.state.data?.status ?? "") || query.state.error ? false : 2000,
  });
  const generate = useMutation({
    mutationFn: () => createLearningReport(scope),
    retry: false,
    onSuccess: (created) => setTaskId(created.task_id),
  });
  const currentReport = task.data?.status === "done" ? task.data.result : report.data;
  const working = generate.isPending || (!!taskId && (!task.data || ["queued", "running"].includes(task.data.status)));
  const error = tab === "mistakes" ? mistakes.error : generate.error || task.error || report.error;
  const errorText = error?.message === "queue_busy" ? (zh ? "任务繁忙，请稍后重试" : "Task queue is busy. Try again shortly.")
    : error?.message === "service_unavailable" ? (zh ? "学习服务暂不可用" : "Learning service unavailable")
    : (zh ? "加载失败，请重试" : "Unable to load. Try again.");

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="drawer-overlay" />
        <Dialog.Content className="inspector-drawer learning-drawer" aria-describedby={undefined}>
          <header className="inspector-header">
            <Dialog.Title>{zh ? "错题与报告" : "Mistakes and reports"}</Dialog.Title>
            <Dialog.Close asChild><button type="button" className="icon-button" aria-label={zh ? "关闭" : "Close"}><X size={18} /></button></Dialog.Close>
          </header>
          <div className="learning-tabs" role="tablist" aria-label={zh ? "学习记录" : "Learning records"}>
            {(["mistakes", "report"] as const).map((item) => (
              <button key={item} type="button" role="tab" aria-selected={tab === item}
                id={`learning-${item}-tab`} aria-controls="learning-content"
                tabIndex={tab === item ? 0 : -1}
                onKeyDown={(event) => {
                  if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                    event.preventDefault();
                    const next = event.key === "Home" ? "mistakes" : event.key === "End" ? "report" : tab === "mistakes" ? "report" : "mistakes";
                    setTab(next);
                    document.getElementById(`learning-${next}-tab`)?.focus();
                  }
                }} onClick={() => setTab(item)}>
                {item === "mistakes" ? <BookOpen size={16} /> : <ChartNoAxesCombined size={16} />}
                {item === "mistakes" ? (zh ? "错题本" : "Mistakes") : (zh ? "学习报告" : "Report")}
              </button>
            ))}
          </div>
          <div id="learning-content" role="tabpanel" aria-labelledby={`learning-${tab}-tab`} className="learning-content">
            {error && <p role="alert" className="learning-error">{errorText}</p>}
            {tab === "mistakes" ? (
              <>
                <div className="learning-toolbar">
                  <span>{zh ? "最近 200 条 · 当前会话" : "Latest 200 · Current session"}</span>
                  <button type="button" className="icon-button" title={zh ? "刷新" : "Refresh"}
                    aria-label={zh ? "刷新错题" : "Refresh mistakes"} disabled={mistakes.isFetching} onClick={() => void mistakes.refetch()}>
                    <RefreshCw size={16} className={mistakes.isFetching ? "spin" : ""} />
                  </button>
                </div>
                {mistakes.isPending && !mistakes.error && <p role="status">{zh ? "加载中" : "Loading"}</p>}
                {mistakes.data?.length === 0 && <p className="source-empty">{zh ? "当前会话还没有错题记录" : "No mistakes in this session yet"}</p>}
                {mistakes.data?.map((item) => (
                  <article className="learning-mistake" key={item.id}>
                    <div className="learning-mistake-meta"><time>{item.created_at}</time><span>{errorLabel(item.error_type, zh)}</span></div>
                    <MarkdownAnswer content={item.question_text} />
                    {!!item.knowledge_points?.length && <p className="learning-points">{item.knowledge_points.join(" / ")}</p>}
                    {item.answer_text && <details><summary>{zh ? "查看解析" : "View explanation"}</summary><MarkdownAnswer content={item.answer_text} /></details>}
                    <button type="button" className="learning-command" disabled={busy} onClick={() => onReview(item.question_text)}>
                      <RefreshCw size={14} />{zh ? "重新解析" : "Review again"}
                    </button>
                  </article>
                ))}
              </>
            ) : (
              <>
                <div className="learning-toolbar">
                  <button type="button" className="learning-command" disabled={working} onClick={() => generate.mutate()}>
                    {working ? <LoaderCircle className="spin" size={16} /> : <ChartNoAxesCombined size={16} />}
                    {working ? (zh ? "生成中" : "Generating") : (zh ? "生成报告" : "Generate report")}
                  </button>
                  <div className="learning-actions">
                    <button type="button" className="icon-button" aria-label={zh ? "刷新报告" : "Refresh report"}
                      title={zh ? "刷新" : "Refresh"} disabled={report.isFetching || task.isFetching}
                      onClick={() => { void report.refetch(); if (taskId) void task.refetch(); }}><RefreshCw size={16} /></button>
                    <button type="button" className="icon-button" aria-label={zh ? "下载报告" : "Download report"}
                      title={zh ? "下载报告" : "Download report"} disabled={!currentReport} onClick={() => currentReport && downloadReport(currentReport)}><Download size={16} /></button>
                  </div>
                </div>
                {taskId && <p role="status" className="learning-task-status">{
                  task.data?.status === "done" ? (zh ? "报告已完成" : "Report ready")
                    : task.data?.status === "failed" ? (zh ? "报告生成失败，可重新生成" : "Report failed. Generate again.")
                    : task.error ? (zh ? "任务状态读取失败，请刷新" : "Unable to read task status. Refresh to retry.")
                    : (zh ? "报告处理中" : "Report in progress")
                }</p>}
                {report.isPending && !report.error && !currentReport && <p role="status">{zh ? "加载中" : "Loading"}</p>}
                {currentReport && <>
                  <dl className="learning-totals"><div><dt>{zh ? "错题总数" : "Total mistakes"}</dt><dd>{currentReport.mistake_count}</dd></div>
                    <div><dt>{zh ? "近期分析题数" : "Recent analysis"}</dt><dd>{currentReport.analyzed_count ?? Math.min(200, currentReport.mistake_count)}</dd></div></dl>
                  {currentReport.mistake_count === 0 ? <p className="source-empty">{zh ? "当前会话暂无可分析的错题" : "No mistakes to analyze in this session"}</p> : <>
                    {([[zh ? "薄弱知识点" : "Knowledge gaps", currentReport.top_knowledge_points], [zh ? "常见错误" : "Frequent errors", currentReport.top_error_types]] as [string, [string, number][]][]).map(([label, entries]) => (
                      <section className="learning-report-section" key={label}><h3>{label}</h3>
                        {entries.length ? <ul>{entries.map(([name, count]) => <li key={name}><span>{errorLabel(name, zh)}</span><strong>{count}</strong></li>)}</ul>
                          : <p className="source-empty">{zh ? "暂无分类" : "Not classified"}</p>}
                      </section>
                    ))}
                    <section className="learning-report-section"><h3>{zh ? "复习建议" : "Review plan"}</h3>
                      <ol>{currentReport.recommendations.map((item) => <li key={item}>{item}</li>)}</ol>
                    </section>
                  </>}
                </>}
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
