import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

# Import search utility from tester script
try:
	from scripts.test_pdf_locator import search_pdf_for_string  # type: ignore
except Exception as e:
	print(f"Failed to import search utility: {e}", file=sys.stderr)
	sys.exit(2)


def _load_json(path: Path) -> Dict[str, Any]:
	try:
		with path.open("r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as e:
		print(f"Failed to read JSON: {path} ({e})", file=sys.stderr)
		sys.exit(2)


def main() -> None:
	ap = argparse.ArgumentParser(description="Batch-run section_start searches from an eval JSON; prints total successes.")
	ap.add_argument("json", type=str, help="Path to page eval JSON (contains pdf, page, and sections array)")
	ap.add_argument("--verbose", action="store_true", help="Print per-section results")
	args = ap.parse_args()

	json_path = Path(args.json)
	if not json_path.exists():
		print(f"JSON not found: {json_path}", file=sys.stderr)
		sys.exit(2)

	obj = _load_json(json_path)
	pdf_name = str(obj.get("pdf") or "").strip()
	page_num = None
	try:
		page_num = int(obj.get("page")) if obj.get("page") is not None else None
	except Exception:
		page_num = None
	sections = obj.get("llm", {}).get("sections") or []
	if not pdf_name or not isinstance(sections, list):
		print("Invalid JSON: missing pdf or sections[]", file=sys.stderr)
		sys.exit(2)

	# Resolve PDF path (default to data/)
	pdf_path = Path(pdf_name)
	if not pdf_path.is_absolute():
		first = (pdf_path.parts[0].lower() if pdf_path.parts else "")
		if first != "data":
			pdf_path = Path("data") / pdf_path
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
		matches = search_pdf_for_string(pdf_path, start, p)
		ok = len(matches) > 0
		if ok:
			success += 1
		if args.verbose:
			print(json.dumps({
				"section_id": it.get("section_id"),
				"page": p,
				"start": start,
				"found": ok,
				"matches": matches,
			}, ensure_ascii=False))

	print(json.dumps({
		"file": pdf_path.name,
		"json": json_path.name,
		"total": total,
		"success": success,
		"rate": (success / total) if total > 0 else 0.0,
	}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
