import { useEffect, useRef, useState } from "react";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  ArrowUp,
  AudioLines,
  FileImage,
  LoaderCircle,
  Paperclip,
  Square,
  X,
} from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { parseAttachment } from "../api";
import { text } from "../i18n";
import type { Language } from "../types";

interface ComposerProps {
  language: Language;
  busy: boolean;
  onSend: (question: string, wrongAnswer: string) => Promise<boolean>;
  onStop: () => void;
}

function IconTip({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip" sideOffset={7}>
          {label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function Composer({
  language,
  busy,
  onSend,
  onStop,
}: ComposerProps) {
  const [question, setQuestion] = useState("");
  const [wrong, setWrong] = useState("");
  const [showWrong, setShowWrong] = useState(false);
  const [voiceActive, setVoiceActive] = useState(false);
  const [attachment, setAttachment] = useState<File | null>(null);
  const [attachmentNote, setAttachmentNote] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  const attachmentMutation = useMutation({
    mutationFn: (file: File) => parseAttachment(file, language),
    onSuccess: (result) => {
      if (result.problem_text) setQuestion(result.problem_text);
      if (result.student_answer) {
        setWrong(result.student_answer);
        setShowWrong(true);
      }
      setAttachmentNote(text(language, "attachmentReady"));
      inputRef.current?.focus();
    },
    onError: () => setAttachmentNote(text(language, "attachmentFailed")),
  });

  useEffect(
    () => () => {
      recognitionRef.current?.stop();
    },
    [],
  );

  const acceptFile = (file: File) => {
    const allowed = [
      "image/jpeg",
      "image/png",
      "image/webp",
      "application/pdf",
    ];
    if (!allowed.includes(file.type) || file.size > 8 * 1024 * 1024) {
      setAttachmentNote(text(language, "invalidFile"));
      return;
    }
    setAttachment(file);
    setAttachmentNote(text(language, "attachmentReading"));
    attachmentMutation.mutate(file);
  };

  const toggleVoice = () => {
    if (voiceActive) {
      recognitionRef.current?.stop();
      return;
    }
    const Recognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!Recognition) {
      setAttachmentNote(
        language === "zh" ? "当前浏览器不支持语音输入" : "Voice input unavailable",
      );
      return;
    }
    const recognition = new Recognition();
    recognition.lang = language === "zh" ? "zh-CN" : "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    const original = question.trim();
    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0]?.transcript || "";
      }
      setQuestion([original, transcript.trim()].filter(Boolean).join(" "));
    };
    recognition.onend = () => setVoiceActive(false);
    recognition.onerror = () => setVoiceActive(false);
    recognitionRef.current = recognition;
    setVoiceActive(true);
    recognition.start();
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const value = question.trim();
    if (!value || busy || attachmentMutation.isPending) return;
    const wrongValue = wrong.trim();
    setQuestion("");
    setWrong("");
    setShowWrong(false);
    setAttachment(null);
    setAttachmentNote("");
    requestAnimationFrame(() => inputRef.current?.focus());
    await onSend(value, wrongValue);
  };

  return (
    <div className="composer-zone">
      <form
        id="askForm"
        className="agent-composer"
        onSubmit={submit}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const file = event.dataTransfer.files[0];
          if (file) acceptFile(file);
        }}
      >
        <AnimatePresence>
          {showWrong && (
            <motion.div
              id="mistakeField"
              className="wrong-input"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <textarea
                id="wrongAnswer"
                value={wrong}
                onChange={(event) => setWrong(event.target.value)}
                placeholder={text(language, "wrongPlaceholder")}
                rows={2}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {(attachment || attachmentNote) && (
          <div className="attachment-chip" id="attachmentPreview">
            {attachmentMutation.isPending ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <FileImage size={16} />
            )}
            <span>{attachment?.name || attachmentNote}</span>
            <small>{attachmentNote}</small>
            <button
              type="button"
              onClick={() => {
                setAttachment(null);
                setAttachmentNote("");
              }}
              aria-label={text(language, "close")}
            >
              <X size={15} />
            </button>
          </div>
        )}

        <textarea
          id="questionInput"
          ref={inputRef}
          value={question}
          rows={1}
          maxLength={8000}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === "Enter" &&
              !event.shiftKey &&
              !event.nativeEvent.isComposing
            ) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          placeholder={text(language, "placeholder")}
          aria-label={text(language, "placeholder")}
        />

        <div className="composer-toolbar">
          <div>
            <Tooltip.Provider delayDuration={300}>
              <IconTip label={text(language, "upload")}>
                <button
                  id="uploadButton"
                  className="composer-tool"
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  disabled={busy}
                >
                  <Paperclip size={18} />
                </button>
              </IconTip>
              <IconTip
                label={
                  voiceActive
                    ? text(language, "listening")
                    : text(language, "voice")
                }
              >
                <button
                  id="voiceButton"
                  className={
                    voiceActive ? "composer-tool recording" : "composer-tool"
                  }
                  type="button"
                  onClick={toggleVoice}
                  disabled={busy}
                >
                  <AudioLines size={18} />
                </button>
              </IconTip>
            </Tooltip.Provider>
            <button
              className={showWrong ? "wrong-toggle active" : "wrong-toggle"}
              type="button"
              onClick={() => setShowWrong((value) => !value)}
            >
              {text(language, "wrongAnswer")}
            </button>
          </div>
          {busy ? (
            <button
              className="send-control stop"
              type="button"
              onClick={onStop}
              id="sendButton"
              aria-label={text(language, "stop")}
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              className="send-control"
              type="submit"
              id="sendButton"
              disabled={!question.trim() || attachmentMutation.isPending}
              aria-label={text(language, "send")}
            >
              <ArrowUp size={19} />
            </button>
          )}
        </div>
        <input
          id="attachmentInput"
          ref={fileRef}
          type="file"
          hidden
          accept="image/jpeg,image/png,image/webp,application/pdf"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) acceptFile(file);
            event.currentTarget.value = "";
          }}
        />
      </form>
    </div>
  );
}
