"use client";

import React, { useMemo, useRef, useState, useEffect } from "react";

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
	const askWrapRef = useRef<HTMLDivElement | null>(null);

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
					<div ref={askWrapRef} style={{ position: 'relative', width: 90 }}>
						<button
							className="btn primary"
							disabled={!selectedGame}
							style={{ fontSize: 18, display: 'grid', alignItems: 'center', lineHeight: 1.1, paddingTop: 6, paddingBottom: 6, width: '100%' }}
							onClick={() => onSubmit()}
						>
							<span>Ask</span>
						</button>
					</div>
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

// Keyframes for slide-up animation injected once
if (typeof document !== 'undefined' && !document.getElementById('style-slideup-keyframes')) {
	const styleEl = document.createElement('style');
	styleEl.id = 'style-slideup-keyframes';
	styleEl.textContent = `@keyframes styleSlideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }`;
	document.head.appendChild(styleEl);
}


 