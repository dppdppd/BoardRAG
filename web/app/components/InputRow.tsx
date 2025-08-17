"use client";

import React from "react";

type Props = {
  isStreaming: boolean;
  input: string;
  onChangeInput: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  toggleSheet: () => void;
  selectedGame?: string | null;
};

export default function InputRow({ isStreaming, input, onChangeInput, onSubmit, onStop, toggleSheet, selectedGame }: Props) {
  return (
    <div className="input-row">
      {isStreaming ? (
        <div className="input indicator" aria-live="polite">
          <span className="spinner" />
          <span style={{ marginLeft: 8 }}>Generating answer…</span>
        </div>
      ) : (
        <input
          className="input"
          placeholder="Your question…"
          value={input}
          onChange={(e) => onChangeInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
        />
      )}

      <div className="actions" style={{ display: "flex", gap: 8, alignItems: "stretch" }}>
        {isStreaming ? (
          <button className="btn stop" onClick={onStop}>Stop</button>
        ) : (
          <button className="btn primary" onClick={onSubmit} disabled={!selectedGame} style={{ fontSize: 18 }}>
            Ask
          </button>
        )}
        <button
          className="btn menu-toggle"
          onClick={toggleSheet}
          aria-label="Menu"
          title="Menu"
          style={{
            display: 'inline-grid',
            placeItems: 'center',
            width: 44,
            height: 44,
            minHeight: 44,
            padding: 0,
            lineHeight: 1,
            fontSize: 20
          }}
        >
          ☰
        </button>
      </div>
    </div>
  );
}


