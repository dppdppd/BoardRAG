from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


SCHEMA_VERSION = "v2.3"


@dataclass
class VisualDesc:
    type: str
    description: str
    relevance: str


@dataclass
class PageChunk:
    id: str
    source: str
    page: int
    next_page: Optional[int]
    full_text: str
    summary: str
    sections: List[str]
    visuals: List[VisualDesc] = field(default_factory=list)
    visual_importance: int = 1
    pdf_sha256: Optional[str] = None
    page_pdf_sha256: Optional[str] = None
    created_at: Optional[str] = None
    version: str = SCHEMA_VERSION

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "page": self.page,
            "next_page": self.next_page,
            "summary": self.summary,
            "sections": self.sections,
            "visuals": [v.__dict__ for v in (self.visuals or [])],
            "visual_importance": self.visual_importance,
            "pdf_sha256": self.pdf_sha256,
            "page_pdf_sha256": self.page_pdf_sha256,
            "created_at": self.created_at,
            "version": self.version,
        }


