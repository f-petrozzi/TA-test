import json
import os
import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Protocol
from urllib.parse import urlparse

import requests

from utils.azure_llm import stream_chat
from utils.supabase_client import get_supabase_client

from dotenv import load_dotenv
load_dotenv(override=True)

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "google/embeddinggemma-300m")

AZURE_ORCHESTRATOR_DEPLOYMENT = os.getenv("AZURE_PHI4_ORCHESTRATOR") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

SUPABASE_MATCH_FUNCTION = os.getenv("SUPABASE_MATCH_FUNCTION", "match_document_chunks")
SUPABASE_DEFAULT_MATCH_COUNT = int(os.getenv("SUPABASE_MATCH_COUNT", "6"))

class MCPClientProtocol(Protocol):
    def retrieve_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        extra_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        ...

def get_system_prompt() -> str:
    text = os.getenv("RAG_SYSTEM_PROMPT")
    if not text:
        raise RuntimeError("Missing required environment variable: RAG_SYSTEM_PROMPT")
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:
        return text

def require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# Token estimator
_WORD_OR_PUNC = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_OR_PUNC.findall(str(text)))

def embed_query(text: str) -> List[float]:
    token = require_env(HUGGINGFACEHUB_API_TOKEN, "HUGGINGFACEHUB_API_TOKEN")
    model = require_env(HUGGINGFACE_MODEL, "HUGGINGFACE_EMBEDDING_MODEL")
    url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"Hugging Face error ({resp.status_code}): {resp.text}")
    data = resp.json()
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Unexpected Hugging Face response: {data}")
    first = data[0]
    vec = first.get("embedding") if isinstance(first, dict) else first
    if not isinstance(vec, list):
        raise RuntimeError(f"Invalid embedding payload: {first}")
    return [float(x) for x in vec]

def retrieve_matches(
    query: str,
    match_count: Optional[int] = None,
    extra_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    client = get_supabase_client()
    embedding = embed_query(query)
    payload: Dict[str, Any] = {
        "query_embedding": embedding,
        "match_count": match_count or SUPABASE_DEFAULT_MATCH_COUNT,
    }
    filter_env = os.getenv("SUPABASE_MATCH_FILTER")
    filt = extra_filter or {}
    if filter_env:
        try:
            filt.update(json.loads(filter_env))
        except json.JSONDecodeError:
            pass
    if filt:
        payload["filter"] = filt
    resp = client.rpc(SUPABASE_MATCH_FUNCTION, payload).execute()
    data = getattr(resp, "data", []) or []
    hits: List[Dict[str, Any]] = []
    for item in data:
        meta = item.get("metadata") or {}
        doc = item.get("content") or item.get("chunk") or meta.get("content") or ""
        sim = item.get("similarity") or item.get("score")
        hits.append(
            {
                "doc": doc,
                "meta": {
                    **meta,
                    "section_title": item.get("section_title") or meta.get("section_title"),
                    "filename": item.get("filename") or meta.get("filename"),
                    "category": item.get("category") or meta.get("category"),
                    "canonical": item.get("canonical") or meta.get("canonical"),
                    "chunk_id": item.get("id") or meta.get("id"),
                },
                "score": sim,
            }
        )
    return hits

def format_context(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "No relevant context found."
    blocks = []
    for i, h in enumerate(hits[:5], 1):
        meta = h.get("meta") or {}
        title = meta.get("section_title") or meta.get("title") or meta.get("filename") or "Section"
        blocks.append(f"Source {i}: {title}\n{h.get('doc', '')}")
    return "\n\n---\n\n".join(blocks)

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""

    def short_url(u: str) -> str:
        if not u or not isinstance(u, str) or not u.startswith("http"):
            return ""
        p = urlparse(u)
        disp = (p.netloc + p.path).rstrip("/")
        return disp if len(disp) <= 80 else disp[:77] + "…"

    lines, seen = [], set()
    i = 0
    for h in hits:
        m = h.get("meta") or {}
        title = m.get("section_title") or m.get("filename") or "Untitled"
        cat = m.get("category") or "Orientation"
        file = m.get("filename") or ""
        canon = (m.get("canonical") or "").strip()

        key = (title, canon)
        if key in seen:
            continue
        seen.add(key)

        i += 1
        suffix = f" · {file}" if file else ""
        if canon:
            disp = short_url(canon)
            lines.append(f"{i}. [{title}]({canon}) — {cat}{suffix}" + (f"\n    ↳ {disp}" if disp else ""))
        else:
            lines.append(f"{i}. {title} — {cat}{suffix}")

        if i >= 5:
            break

    return "\n".join(lines)

def _augment_query(user_text: str) -> str:
    lowered = (user_text or "").lower()
    keywords: List[str] = []

    if "orientation" in lowered:
        keywords.extend([
            "orientation",
            "orientation dates",
            "orientation fees",
            "orientation schedule",
            "myorientation",
        ])
        if "international" in lowered:
            keywords.extend([
                "international orientation",
                "glo-bull beginnings",
                "international student orientation",
                "mybullspath",
            ])
        if any(term in lowered for term in ("freshman", "first-year", "ftic")):
            keywords.append("first-year orientation")

    if "international" in lowered and "orientation" not in lowered:
        keywords.extend([
            "international student services",
            "glo-bull beginnings",
        ])

    if keywords:
        dedup = []
        seen = set()
        for word in keywords:
            if word not in seen:
                dedup.append(word)
                seen.add(word)
        return f"{user_text}\n\nRelated keywords: {', '.join(dedup)}"
    return user_text


def generate_with_rag(
    user_text: str,
    match_count: Optional[int] = None,
    mcp_client: Optional[MCPClientProtocol] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    query = _augment_query(user_text)
    if mcp_client:
        hits = mcp_client.retrieve_context(query, match_count=match_count)
    else:
        hits = retrieve_matches(query, match_count=match_count)
    system = get_system_prompt()
    ctx = format_context(hits)

    messages = [
        {"role": "system", "content": f"{system}\n\nCONTEXT:\n{ctx}"},
        {"role": "user", "content": user_text},
    ]

    response_text = ""
    for delta in stream_chat(AZURE_ORCHESTRATOR_DEPLOYMENT, messages):
        response_text += delta
        yield ("delta", {"text": response_text})

    sources_block = build_sources_block(hits)
    final = (response_text or "").rstrip()
    if sources_block:
        final += "\n\n**Sources**\n" + sources_block
    yield ("final", {"text": final, "hits": hits})
