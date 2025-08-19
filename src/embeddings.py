from __future__ import annotations

from typing import List


def embed_texts(texts: List[str]) -> List[List[float]]:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    from . import config as cfg  # type: ignore

    model = getattr(cfg, "EMBEDDER_MODEL", "text-embedding-3-small")
    emb = OpenAIEmbeddings(model=model)
    return emb.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    from . import config as cfg  # type: ignore

    model = getattr(cfg, "EMBEDDER_MODEL", "text-embedding-3-small")
    emb = OpenAIEmbeddings(model=model)
    return emb.embed_query(text)


