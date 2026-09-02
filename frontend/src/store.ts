import { create } from "zustand";
import { persist } from "zustand/middleware";
import type {
  AskResponse,
  ChatMessage,
  ChatSession,
  Language,
  Theme,
} from "./types";

const id = () => crypto.randomUUID();

function emptySession(): ChatSession {
  return {
    id: id(),
    title: "新订正",
    messages: [],
    history: [],
    summary: "",
    exercise: null,
    updatedAt: Date.now(),
  };
}

interface ChatStore {
  sessions: ChatSession[];
  activeId: string;
  language: Language;
  theme: Theme;
  createSession: () => string;
  selectSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  addMessage: (message: ChatMessage) => void;
  replaceLastAssistant: (message: ChatMessage) => void;
  applyResponse: (response: AskResponse) => void;
  setLanguage: (language: Language) => void;
  setTheme: (theme: Theme) => void;
}

const initial = emptySession();

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      sessions: [initial],
      activeId: initial.id,
      language: "zh",
      theme: "light",
      createSession: () => {
        const session = emptySession();
        set((state) => ({
          sessions: [session, ...state.sessions],
          activeId: session.id,
        }));
        return session.id;
      },
      selectSession: (sessionId) => set({ activeId: sessionId }),
      deleteSession: (sessionId) => {
        const remaining = get().sessions.filter((item) => item.id !== sessionId);
        const sessions = remaining.length ? remaining : [emptySession()];
        set((state) => ({
          sessions,
          activeId:
            state.activeId === sessionId ? sessions[0].id : state.activeId,
        }));
      },
      addMessage: (message) =>
        set((state) => ({
          sessions: state.sessions.map((session) => {
            if (session.id !== state.activeId) return session;
            const firstUser =
              message.role === "user" &&
              !session.messages.some((item) => item.role === "user");
            return {
              ...session,
              title: firstUser
                ? message.content.replace(/\s+/g, " ").slice(0, 28)
                : session.title,
              messages: [...session.messages, message],
              updatedAt: Date.now(),
            };
          }),
        })),
      replaceLastAssistant: (message) =>
        set((state) => ({
          sessions: state.sessions.map((session) => {
            if (session.id !== state.activeId) return session;
            const messages = [...session.messages];
            const index = messages.findLastIndex(
              (item) => item.role === "assistant",
            );
            if (index >= 0) messages[index] = message;
            else messages.push(message);
            return { ...session, messages, updatedAt: Date.now() };
          }),
        })),
      applyResponse: (response) =>
        set((state) => ({
          sessions: state.sessions.map((session) =>
            session.id === state.activeId
              ? {
                  ...session,
                  history: response.conversation_history || session.history,
                  summary: response.conversation_summary || "",
                  exercise: response.exercise_state ?? null,
                  updatedAt: Date.now(),
                }
              : session,
          ),
        })),
      setLanguage: (language) => set({ language }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: "mathtrace-agent-workspace-v2",
      partialize: (state) => ({
        sessions: state.sessions.slice(0, 30),
        activeId: state.activeId,
        language: state.language,
        theme: state.theme,
      }),
    },
  ),
);

export const newMessage = (
  role: ChatMessage["role"],
  content: string,
  extra: Partial<ChatMessage> = {},
): ChatMessage => ({
  id: id(),
  role,
  content,
  createdAt: Date.now(),
  ...extra,
});
