import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

# Config and helpers
try:
	from src import config as cfg  # type: ignore
except Exception:
	cfg = None  # type: ignore

# Use app locator for parity with production
try:
	from src.pdf_utils import compute_normalized_section_start_bbox_exact as _loc  # type: ignore
	from src.pdf_pages import page_slug_from_pdf  # type: ignore
except Exception as e:
	print(f"Failed to import app locator: {e}", file=sys.stderr)
	sys.exit(2)


def _load_json(path: Path) -> Dict[str, Any]:
	try:
		with path.open("r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as e:
		print(f"Failed to read JSON: {path} ({e})", file=sys.stderr)
		sys.exit(2)


def main() -> None:
	ap = argparse.ArgumentParser(description="Batch-run section_start searches; infer JSON from --pdf/--page if not provided.")
	ap.add_argument("--json", type=str, help="Path to page eval JSON (optional if --pdf and --page provided)")
	ap.add_argument("--pdf", type=str, help="Base PDF filename or path (relative to data/ allowed)")
	ap.add_argument("--page", type=int, help="1-based page number within the base PDF")
	ap.add_argument("--verbose", action="store_true", help="Print per-section results")
	args = ap.parse_args()

	json_path: Optional[Path] = None
	if getattr(args, "json", None):
		json_path = Path(str(args.json))
		if not json_path.exists():
			print(f"JSON not found: {json_path}", file=sys.stderr)
			sys.exit(2)
	else:
		# Require pdf and page to infer JSON path
		if not getattr(args, "pdf", None) or getattr(args, "page", None) is None:
			print("Missing inputs: provide --json or both --pdf and --page", file=sys.stderr)
			sys.exit(2)
		# Resolve DATA_PATH (default to data/)
		from pathlib import Path as _P
		data_dir = _P(getattr(cfg, "DATA_PATH", "data")) if cfg is not None else _P("data")
		pdf_arg = _P(str(args.pdf)).name if not _P(str(args.pdf)).is_absolute() else _P(str(args.pdf)).name
		# Build eval JSON path: data/<stem>/3_eval_jsons/<slug>_pNNNN.json
		stem = _P(pdf_arg).stem
		slug = page_slug_from_pdf(_P(pdf_arg))
		page_num = max(1, int(args.page))
		eval_dir = data_dir / stem / "3_eval_jsons"
		json_path = eval_dir / f"{slug}_p{page_num:04}.json"
		if not json_path.exists():
			print(f"Eval JSON not found: {json_path}", file=sys.stderr)
			sys.exit(2)

	obj = _load_json(json_path)
	# Allow CLI overrides for pdf and page
	pdf_name = str(args.pdf).strip() if getattr(args, "pdf", None) else str(obj.get("pdf") or "").strip()
	page_num: Optional[int] = None
	if getattr(args, "page", None) is not None:
		try:
			page_num = int(args.page)
		except Exception:
			page_num = None
	else:
		try:
			page_num = int(obj.get("page")) if obj.get("page") is not None else None
		except Exception:
			page_num = None
	sections = obj.get("llm", {}).get("sections") or []
	if not isinstance(sections, list):
		print("Invalid JSON: sections[] missing or not a list", file=sys.stderr)
		sys.exit(2)
	if not pdf_name:
		print("Missing base PDF name/path: provide --pdf or include 'pdf' in JSON", file=sys.stderr)
		sys.exit(2)

	# Resolve PDF path (default to data/)
	pdf_path = Path(pdf_name)
	if not pdf_path.is_absolute():
		first = (pdf_path.parts[0].lower() if pdf_path.parts else "")
		if first != "data":
			# Prefer configured DATA_PATH if available
			base_data = Path(getattr(cfg, "DATA_PATH", "data")) if cfg is not None else Path("data")
			pdf_path = base_data / pdf_path
	if not pdf_path.exists():
		print(f"PDF not found: {pdf_path}", file=sys.stderr)
		sys.exit(2)

	total = 0
	success = 0
	for it in sections:
		if not isinstance(it, dict):
			continue
		start = str(it.get("section_start") or "").strip()
		p = None
		try:
			p = int(it.get("page")) if it.get("page") is not None else page_num
		except Exception:
			p = page_num
		if not start:
			continue
		total += 1
		bbox = _loc(str(pdf_path), int(p) if p else 1, start) if p else _loc(str(pdf_path), 1, start)
		ok = bool(bbox)
		if ok:
			success += 1
		if args.verbose:
			payload = {
				"section_id": it.get("section_id"),
				"page": p,
				"start": start,
				"found": ok,
			}
			if bbox:
				payload["bbox_pct"] = bbox
			print(json.dumps(payload, ensure_ascii=False))

	print(json.dumps({
		"file": pdf_path.name,
		"json": json_path.name,
		"total": total,
		"success": success,
		"rate": (success / total) if total > 0 else 0.0,
	}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
