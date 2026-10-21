import { useMemo, useState } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  BookOpen,
  MoreHorizontal,
  PanelLeftClose,
  Plus,
  Search,
  Trash2,
} from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { text } from "../i18n";
import { useChatStore } from "../store";

interface SidebarProps {
  open: boolean;
  busy: boolean;
  onClose: () => void;
}

export function Sidebar({ open, busy, onClose }: SidebarProps) {
  const [search, setSearch] = useState("");
  const {
    sessions,
    activeId,
    language,
    createSession,
    selectSession,
    deleteSession,
  } = useChatStore();
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return [...sessions]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .filter((session) => !query || session.title.toLowerCase().includes(query));
  }, [search, sessions]);

  const startNew = () => {
    if (busy) return;
    createSession();
    onClose();
  };

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.button
            aria-label={text(language, "close")}
            className="sidebar-scrim"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
        )}
      </AnimatePresence>
      <aside className={open ? "app-sidebar open" : "app-sidebar"} id="sidebar">
        <div className="brand-row">
          <div className="brand-symbol" aria-hidden="true">
            <span>Σ</span>
            <i />
          </div>
          <div className="brand-copy">
            <strong>{text(language, "brand")}</strong>
            <span>{text(language, "subtitle")}</span>
          </div>
          <button
            className="icon-button sidebar-close"
            type="button"
            onClick={onClose}
            aria-label={text(language, "close")}
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button
          id="newChatButton"
          className="new-chat"
          type="button"
          onClick={startNew}
          disabled={busy}
        >
          <Plus size={17} />
          <span>{text(language, "newChat")}</span>
        </button>

        <label className="history-search">
          <Search size={15} aria-hidden="true" />
          <span className="sr-only">{text(language, "search")}</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={text(language, "search")}
          />
        </label>

        <div className="history-heading">{text(language, "today")}</div>
        <nav className="session-list" aria-label={text(language, "today")}>
          {filtered.length === 0 && (
            <p className="history-empty">{text(language, "emptyHistory")}</p>
          )}
          {filtered.map((session) => (
            <div
              className={
                session.id === activeId ? "session-row active" : "session-row"
              }
              key={session.id}
            >
              <button
                className="session-main"
                type="button"
                disabled={busy}
                onClick={() => {
                  selectSession(session.id);
                  onClose();
                }}
              >
                <BookOpen size={15} />
                <span>{session.title}</span>
              </button>
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <button
                    className="session-menu"
                    type="button"
                    aria-label={text(language, "delete")}
                    disabled={busy}
                  >
                    <MoreHorizontal size={16} />
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content className="context-menu" sideOffset={6}>
                    <DropdownMenu.Item
                      className="context-menu-item danger"
                      onSelect={() => deleteSession(session.id)}
                    >
                      <Trash2 size={15} />
                      {text(language, "delete")}
                    </DropdownMenu.Item>
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
            </div>
          ))}
        </nav>

      </aside>
    </>
  );
}
