from __future__ import annotations

from typing import List

from langchain.schema.document import Document

from ..config import (
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_RESULTS,
    SEARCH_PROVIDER,
    SERPAPI_API_KEY,
    BRAVE_API_KEY,
    ENABLE_SEARCH_REWRITE,
)
from .. import config as cfg


def perform_web_search(query: str, k: int = 5) -> List[Document]:
    if not ENABLE_WEB_SEARCH:
        return []

    results = []

    if SEARCH_PROVIDER == "serpapi":
        if not SERPAPI_API_KEY:
            print("⚠️ SERPAPI_API_KEY not set; falling back to DuckDuckGo")
        else:
            try:
                from serpapi import GoogleSearch  # type: ignore

                params = {"q": query, "api_key": SERPAPI_API_KEY, "num": k, "engine": "google"}
                search = GoogleSearch(params)
                serp_results = search.get_dict()
                organic = serp_results.get("organic_results", [])
                for item in organic[:k]:
                    results.append({"snippet": item.get("snippet", ""), "url": item.get("link", "")})
            except Exception as e:
                print(f"⚠️ SerpAPI search failed: {e}")

    elif SEARCH_PROVIDER == "brave":
        if not BRAVE_API_KEY:
            print("⚠️ BRAVE_API_KEY not set; falling back to DuckDuckGo")
        else:
            try:
                import requests

                endpoint = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": BRAVE_API_KEY}
                params = {"q": query, "count": k}
                resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("web", {}).get("results", [])[:k]:
                        results.append({"snippet": item.get("description", ""), "url": item.get("url", "")})
                else:
                    print(f"⚠️ Brave API HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as e:
                print(f"⚠️ Brave search failed: {e}")

    # DuckDuckGo or fallback
    if not results and SEARCH_PROVIDER in {"duckduckgo", "serpapi", "brave"}:
        try:
            from duckduckgo_search import ddg  # type: ignore

            results = ddg(query, max_results=k) or []
        except Exception:
            try:
                from duckduckgo_search import DDGS  # type: ignore

                with DDGS() as search:
                    results = search.text(query, max_results=k) or []
            except Exception as e:
                print(f"⚠️ DDG search failed: {e}")

    if not results:
        print("⚠️ Web search returned 0 results")
        return []

    docs: List[Document] = []
    for res in results:
        snippet = res.get("snippet") or res.get("body") or res.get("text") or res.get("title") or ""
        url = res.get("url") or res.get("href") or res.get("link") or ""
        if not snippet or not url:
            continue
        meta = {"id": url, "source": url, "url": url, "web": True}
        docs.append(Document(page_content=snippet, metadata=meta))
    print(f"🌐 Added {len(docs)} web snippets to context")
    return docs


def rewrite_search_query(raw_query: str) -> str:
    if not ENABLE_SEARCH_REWRITE:
        return raw_query

    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_openai import ChatOpenAI

        print("✏️  Rewriting web search query via LLM …")
        if cfg.LLM_PROVIDER.lower() == "anthropic":
            model = ChatAnthropic(model=cfg.SEARCH_REWRITE_MODEL, temperature=0, max_tokens=512)
        elif cfg.LLM_PROVIDER.lower() == "ollama":
            from langchain_community.llms.ollama import Ollama

            model = Ollama(model=cfg.SEARCH_REWRITE_MODEL, base_url=cfg.OLLAMA_URL)
        else:
            model = ChatOpenAI(model=cfg.SEARCH_REWRITE_MODEL, temperature=0, timeout=60)

        prompt = (
            "You are a search expert. Rewrite the following user question into a concise, "
            "effective web search query. Use quotation marks around exact phrases only if "
            "they are essential. Remove polite fluff. Return one line only, no extra text.\n\n"
            f"User question: {raw_query}\n\nSearch query:"
        )
        rewritten = model.invoke(prompt)
        rewritten_text = getattr(rewritten, "content", str(rewritten)).strip()
        if 3 <= len(rewritten_text) <= 200:
            print(f"✏️  Rewritten query: {rewritten_text!r}")
            return rewritten_text
    except Exception as e:
        print(f"⚠️ Query rewrite failed: {e}")
    return raw_query


