"use client";

import React from "react";
import MarkdownWithSectionLinks from "./MarkdownWithSectionLinks";

type Message = { role: "user" | "assistant"; content: string; pinned?: boolean };

type Props = {
  messages: Message[];
  chatEndRef: React.RefObject<HTMLDivElement>;
  retryableUsers: number[];
  longPressDeleteAssistant: (assistantIndex: number) => any;
  longPressHandlers: (assistantIndex: number) => any;
  togglePin: (assistantIndex: number) => void;
  onRetry: (userIndex: number, content: string) => void;
  onDeleteUser: (userIndex: number) => void;
  openSectionModal: (section: string) => void;
  selectedGame?: string | "";
  setRetryableUsers: React.Dispatch<React.SetStateAction<number[]>>;
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  sessionId: string;
  decorateCitations?: (s: string) => string;
};

export default function ChatMessageList({ messages, chatEndRef, retryableUsers, longPressDeleteAssistant, longPressHandlers, togglePin, onRetry, onDeleteUser, openSectionModal, selectedGame, setRetryableUsers, setMessages, sessionId, decorateCitations }: Props) {
  return (
    <div className="chat-scroll">
      {(() => {
        let assistantCounter = -1;
        let userCounter = -1;
        return messages.map((m, i) => {
          const props: any = { key: i, className: `bubble ${m.role}` };
          let acForHandlers: number | null = null;
          let ucForHandlers: number | null = null;
          if (m.role === "assistant") {
            assistantCounter += 1;
            props["data-assistant-index"] = assistantCounter;
            acForHandlers = assistantCounter;
          } else if (m.role === "user") {
            userCounter += 1;
            props["data-user-index"] = userCounter;
            ucForHandlers = userCounter;
          }
          const isUser = m.role === "user";
          const showActions = isUser && ucForHandlers != null && retryableUsers.includes(ucForHandlers);
          return (
            <React.Fragment key={i}>
              {m.role === 'assistant' ? (
                <div className="assistant-row" style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto', columnGap: 6, alignItems: 'end' }}>
                  <div {...props} style={{ ...(props.style || {}), maxWidth: '100%' }}>
                    <MarkdownWithSectionLinks
                      content={decorateCitations ? decorateCitations(m.content) : m.content}
                      onSectionClick={(sec) => openSectionModal(sec)}
                    />
                  </div>
                  <div className="assistant-actions" style={{ display: 'flex', flexDirection: 'column', gap: 6, alignSelf: 'end' }}>
                    <button
                      className="btn"
                      title={m.pinned ? "Remove bookmark" : "Add bookmark"}
                      aria-label={m.pinned ? "Remove bookmark" : "Add bookmark"}
                      aria-pressed={!!m.pinned}
                      onClick={() => acForHandlers != null && togglePin(acForHandlers)}
                      style={{ width: 36, height: 36, minHeight: 36, padding: 0, display: 'inline-grid', placeItems: 'center', color: m.pinned ? '#fff' : 'var(--text)', borderColor: m.pinned ? 'var(--accent)' : 'var(--control-border)', background: m.pinned ? 'var(--accent)' : 'var(--control-bg)' }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <path d="M6 2h8a2 2 0 0 1 2 2v16l-6-4-6 4V4a2 2 0 0 1 2-2z" fill={m.pinned ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
                      </svg>
                    </button>
                    <button
                      className="btn"
                      title="Hold to delete"
                      aria-label="Delete QA (hold)"
                      {...(acForHandlers != null ? longPressDeleteAssistant(acForHandlers) : {})}
                      style={{ width: 36, height: 36, minHeight: 36, padding: 0, display: 'inline-grid', placeItems: 'center', background: 'var(--control-bg)', borderColor: 'var(--control-border)' }}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                        <path d="M3 6h18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                        <path d="M10 11v7M14 11v7" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      </svg>
                    </button>
                  </div>
                </div>
              ) : (
                <div {...props}>
                  <MarkdownWithSectionLinks
                    content={m.content}
                    onSectionClick={(sec) => {
                      try {
                        if (selectedGame) localStorage.setItem(`boardrag_last_section:${selectedGame}`, sec);
                      } catch {}
                      openSectionModal(sec);
                    }}
                  />
                </div>
              )}
              {showActions && (
                <div style={{ maxWidth: "92%", marginLeft: "auto", marginTop: 6 }}>
                  <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                    <button
                      className="btn"
                      title="Retry"
                      aria-label="Retry"
                      onClick={() => { if (ucForHandlers == null) return; setRetryableUsers((cur) => cur.filter((n) => n !== ucForHandlers!)); onRetry(ucForHandlers!, m.content); }}
                      style={{ width: 44, height: 44, minHeight: 44, padding: 0, fontSize: 18, color: "#fff", borderColor: "var(--accent)", background: "var(--accent)", display: "inline-grid", placeItems: "center" }}
                    >↻</button>
                    <button
                      className="btn"
                      title="Delete"
                      aria-label="Delete"
                      onClick={() => onDeleteUser(ucForHandlers!)}
                      style={{ width: 44, height: 44, minHeight: 44, padding: 0, fontSize: 18, color: "#fff", borderColor: "var(--accent)", background: "var(--accent)", display: "inline-grid", placeItems: "center" }}
                    >✕</button>
                  </div>
                </div>
              )}
            </React.Fragment>
          );
        });
      })()}
      <div ref={chatEndRef} />
    </div>
  );
}


