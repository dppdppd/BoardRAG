"""Watch the `data/` folder for PDF changes and keep the Chroma DB in sync.

Run standalone:
    python watcher.py

Inside Docker this file is launched in background alongside `app.py` (see
docker-compose.yml).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

DATA_DIR = Path(__file__).parent / "data"
PYTHON = sys.executable  # current interpreter inside venv or container


class FileEventHandler(PatternMatchingEventHandler):
    """Rebuild DB whenever a supported document changes."""

    patterns = ["*"]  # watch every file; we'll filter by extension below

    SUPPORTED_EXTS = {".pdf", ".PDF"}  # Extend here when loaders grow

    def on_any_event(self, event):  # noqa: D401
        path = Path(event.src_path)
        if path.suffix not in self.SUPPORTED_EXTS:
            return  # ignore unsupported file types and temp-file noise

        verb = event.event_type  # created, modified, deleted, moved
        print(f"📄  {verb.capitalize()} {path.name}. Rebuilding database …", flush=True)
        subprocess.run([PYTHON, "populate_database.py"], check=True)


def main() -> None:
    if not DATA_DIR.exists():
        print("⚠️  data/ directory not found; watcher exiting", flush=True)
        sys.exit(1)

    handler = FileEventHandler()
    observer = Observer()
    observer.schedule(handler, DATA_DIR, recursive=True)
    observer.start()

    # Trigger an initial build so the DB exists even before changes occur.
    print("🔄  Initial database build …", flush=True)
    subprocess.run([PYTHON, "populate_database.py"], check=True)

    print(f"👀  Watching {DATA_DIR} for PDFs. Press Ctrl-C to stop.", flush=True)
    try:
        while True:
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
