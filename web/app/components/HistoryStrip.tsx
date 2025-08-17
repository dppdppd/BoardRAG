"use client";

import React from "react";

type Props = {
  labels: string[];
  containerRef: React.RefObject<HTMLDivElement>;
  onClickLabel: (userIndex: number) => void;
  longPressHandlersFor: (assistantIndex: number) => any;
  bookmarkUserIndices: number[];
  bookmarkAssistantIndices: number[];
  isDragging: () => boolean;
};

export default function HistoryStrip({ labels, containerRef, onClickLabel, longPressHandlersFor, bookmarkUserIndices, bookmarkAssistantIndices, isDragging }: Props) {
  return (
    <div className="history-strip" ref={containerRef}>
      {labels.length > 0 && (
        <>
          {labels.map((label, i) => (
            <button
              key={i}
              className="history-pill btn"
              onClick={() => {
                if (isDragging()) return;
                onClickLabel(bookmarkUserIndices[i]);
              }}
              {...longPressHandlersFor(bookmarkAssistantIndices[i])}
            >
              {label}
            </button>
          ))}
        </>
      )}
    </div>
  );
}


