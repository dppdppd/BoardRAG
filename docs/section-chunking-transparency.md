### Section chunking: per-PDF schema detection and transparent logging

### Status
- Implemented per-PDF section scheme detection and verbose logging.
- Wired the chunker to use only the best-matching header family per PDF.
- Added a helper to audit all PDFs’ schemes.

### What changed
- The chunker now:
  - Samples the first few pages per PDF.
  - Scores three families: numeric, alphanumeric, word-based.
  - Prints the raw scores and the chosen scheme.
  - Uses only that scheme’s regexes while splitting.
  - Streams every detected section header as it processes pages.

### How to run transparently (Windows)
- Verify HF4 with phrase-focused diagnostic (prints scheme scores + all detected headers while splitting):
```bat
venv\Scripts\python.exe temp_tests\test_chunking_headers.py --pdf "data\HF4 Core Rules.pdf" --phrase "Boost Operation" --max 10
```
What you’ll see near the top:
- “📑 Section scheme detection (by PDF):”
- “HF4 Core Rules.pdf: scores={numeric:…, alphanum:…, words:…} → chosen=alphanum”
Then during processing:
- “• p29 I4 Boost Operation [HF4 Core Rules.pdf]”
- “• p15 F2 Dry Mass Adjustment …”
- “• p16 F3 Wet Mass Adjustment …”
…etc.

- Scan all PDFs to see their inferred schemes:
```bat
venv\Scripts\python.exe temp_tests\detect_section_schemes.py
```

- Rebuild a single PDF into the DB with the new logic and verbose spew:
```bat
venv\Scripts\python.exe -m src.query --query_text "example"
```

- Inspect stored chunks by file (run as module to avoid import-path issues):
```bat
venv\Scripts\python.exe -m temp_tests.inspect_chunks --no-content --filter "HF4 Core Rules.pdf" --max 60
```

### Notes
- HF4 should now consistently choose alphanumeric and emit sections like `E4`, `F3`, `G1`, `I4`, etc. Subsections remain embedded in chunk content (e.g., “I4a”, “I4b” within the same chunk when they don’t begin a new header line).
- Numeric-style games (e.g., ASLSK4) should choose numeric; “words” games (e.g., some family games) will choose word-based headings.

### Next optional enhancement
- LLM-assisted fallback: send 1–2 pages to an LLM when scores are tied/weak. Can be added behind a flag.

### More examples
- Test other phrases quickly:
```bat
venv\Scripts\python.exe temp_tests\test_chunking_headers.py --pdf "data\HF4 Core Rules.pdf" --phrase "Wet Mass Adjustment" --max 10
venv\Scripts\python.exe temp_tests\test_chunking_headers.py --pdf "data\HF4 Core Rules.pdf" --phrase "Cargo Transfer" --max 10
venv\Scripts\python.exe temp_tests\test_chunking_headers.py --pdf "data\HF4 Core Rules.pdf" --phrase "Movement Restrictions" --max 10
```

- DB-wide section list including pages for HF4 (plus a raw PDF scan):
```bat
venv\Scripts\python.exe temp_tests\list_sections.py --pdf "HF4 Core Rules.pdf" --scan-pdf --sort page --max 500
```

- Fetch chunks that include a phrase (e.g., “Boost Operation”) from the DB:
```bat
venv\Scripts\python.exe temp_tests\find_section.py --pdf "HF4 Core Rules.pdf" --contains "Boost Operation" --max 20
```

- View chunks by a section tag (simulate UI “[I4]” click):
```bat
venv\Scripts\python.exe temp_tests\debug_section_chunks.py --section I4 --game "High Frontier 4 All" --limit 12
```

- Adjust sampling window for scheme detection: detector samples up to 8 pages by default (see `src/section_schemas.py`).
- To force a specific scheme for a given PDF, update `KNOWN_PDF_SCHEMES` in `src/section_schemas.py`.


