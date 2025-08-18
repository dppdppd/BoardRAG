"use client";

import React, { useMemo, useRef, useState, useEffect } from "react";

type PromptStyle = "default" | "brief" | "detailed";

type Props = {
	isStreaming: boolean;
	input: string;
	onChangeInput: (value: string) => void;
	onSubmit: () => void;
	onStop: () => void;
	toggleSheet: () => void;
	selectedGame?: string | null;
	promptStyle: PromptStyle;
	setPromptStyle: (s: PromptStyle) => void;
	onSubmitWithStyle?: (s: PromptStyle) => void;
};

export default function InputRow({ isStreaming, input, onChangeInput, onSubmit, onStop, toggleSheet, selectedGame, promptStyle, setPromptStyle, onSubmitWithStyle }: Props) {
	const [showStylePicker, setShowStylePicker] = useState<boolean>(false);
	const pressTimerRef = useRef<any>(null);
	const longPressTriggeredRef = useRef<boolean>(false);
	const askWrapRef = useRef<HTMLDivElement | null>(null);

	const otherStyles = useMemo<PromptStyle[]>(() => {
		const all: PromptStyle[] = ["default", "brief", "detailed"];
		return all.filter((s) => s !== promptStyle);
	}, [promptStyle]);

	const startLongPress = () => {
		if (!selectedGame) return;
		longPressTriggeredRef.current = false;
		if (pressTimerRef.current) clearTimeout(pressTimerRef.current);
		pressTimerRef.current = setTimeout(() => {
			longPressTriggeredRef.current = true;
			setShowStylePicker(true);
		}, 600);
	};
	const clearLongPress = () => {
		if (pressTimerRef.current) {
			clearTimeout(pressTimerRef.current);
			pressTimerRef.current = null;
		}
	};

	const friendly = (s: PromptStyle): string => (s === "default" ? "normal" : s);

	useEffect(() => {
		if (!showStylePicker) return;
		const onDocPointer = (e: any) => {
			const root = askWrapRef.current;
			if (!root) return setShowStylePicker(false);
			if (!root.contains(e.target)) setShowStylePicker(false);
		};
		document.addEventListener('mousedown', onDocPointer, true);
		document.addEventListener('touchstart', onDocPointer, true);
		return () => {
			document.removeEventListener('mousedown', onDocPointer, true);
			document.removeEventListener('touchstart', onDocPointer, true);
		};
	}, [showStylePicker]);

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
						{/* Ask button with subtle style label */}
						<button
							className="btn primary"
							disabled={!selectedGame}
							style={{ fontSize: 18, display: 'grid', gridTemplateRows: '1fr auto', alignItems: 'center', lineHeight: 1.1, paddingTop: 6, paddingBottom: 6, width: '100%' }}
							onMouseDown={startLongPress}
							onMouseUp={(e) => { const wasTriggered = longPressTriggeredRef.current; clearLongPress(); if (!wasTriggered) onSubmit(); }}
							onMouseLeave={clearLongPress}
							onTouchStart={(e) => { e.preventDefault(); startLongPress(); }}
							onTouchEnd={(e) => { e.preventDefault(); const wasTriggered = longPressTriggeredRef.current; clearLongPress(); if (!wasTriggered) onSubmit(); }}
						>
							<span>Ask</span>
							<span style={{ fontSize: 11, opacity: 0.7, alignSelf: 'end' }} aria-label={`Style: ${friendly(promptStyle)}`}>
								{friendly(promptStyle)}
							</span>
						</button>

						{/* Slide-up style options on long-press (single vertical panel) */}
						{showStylePicker && !isStreaming && selectedGame && (
							<div
								role="menu"
								style={{
									position: 'absolute',
									bottom: '100%',
									left: 0,
									right: 0,
									paddingBottom: 6,
									zIndex: 20,
									pointerEvents: 'auto',
									transform: 'translateY(6px)',
									animation: 'styleSlideUp 160ms ease-out forwards',
									display: 'grid',
									placeItems: 'center',
								}}
							>
								<div
									style={{
										display: 'flex',
										flexDirection: 'column',
										gap: 6,
										background: 'var(--surface)',
										border: '1px solid var(--control-border)',
										borderRadius: 'var(--radius)',
										boxShadow: '0 6px 16px rgba(0,0,0,.15)'
									}}
								>
									{otherStyles.map((s) => (
										<button
											key={s}
											className="btn"
											style={{
												padding: '8px 12px',
												fontSize: 13,
												lineHeight: 1.1,
												background: 'var(--control-bg)',
												borderColor: 'var(--control-border)',
												borderRadius: 0,
												textAlign: 'left',
												minWidth: 160
											}}
											onClick={() => {
												setShowStylePicker(false);
												if (onSubmitWithStyle) {
													onSubmitWithStyle(s);
													setPromptStyle(s);
												} else {
													setPromptStyle(s);
													setTimeout(() => onSubmit(), 0);
												}
											}}
										>
											{friendly(s)}
										</button>
									))}
								</div>
							</div>
						)}
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


 