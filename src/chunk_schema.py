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



@dataclass
class SectionChunk:
    id: str
    source: str
    section_code: str
    section_id2: Optional[str]
    title: str
    section_start: str
    first_page: Optional[int]
    pages: List[int] = field(default_factory=list)
    summary: str = ""
    visuals: List[VisualDesc] = field(default_factory=list)
    visual_importance: int = 1
    created_at: Optional[str] = None
    version: str = SCHEMA_VERSION

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "section_code": self.section_code,
            "section_id2": self.section_id2,
            "section_title": self.title,
            "section_start": self.section_start,
            "first_page": self.first_page,
            "pages": self.pages,
            "summary": self.summary,
            "visuals": [v.__dict__ for v in (self.visuals or [])],
            "visual_importance": self.visual_importance,
            "version": self.version,
            "created_at": self.created_at,
            # Identify type for downstream conditionals
            "chunk_kind": "section",
        }
