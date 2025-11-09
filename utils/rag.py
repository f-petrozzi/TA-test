import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Protocol, TYPE_CHECKING
from urllib.parse import urlparse

import requests
from openai import OpenAI

from utils.supabase_client import get_supabase_client

from dotenv import load_dotenv
load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = (
    "You are the USF Onboarding Assistant for Admissions, Orientation, and Registrar. "
    "Answer ONLY from the provided CONTEXT. Be concise. Add inline [Source N] markers "
    "that match the numbered sources in CONTEXT. If an answer is not in CONTEXT, say "
    "you don't know and suggest the correct USF office or link to contact."
)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "google/embeddinggemma-300m")

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
    if text:
        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        return text
    return DEFAULT_SYSTEM_PROMPT

def require_env(value: Optional[str], name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

@lru_cache(maxsize=1)
def get_azure_client() -> OpenAI:
    endpoint = require_env(AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT")
    key = require_env(AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY")
    # Normalize to the example’s pattern
    base = endpoint.rstrip("/") + "/"
    if not base.endswith("openai/v1/"):
        base += "openai/v1/"
    return OpenAI(base_url=base, api_key=key)

# Token estimator
_WORD_OR_PUNC = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def estimate_tokens(text: str) -> int:
    """
    Simple, dependency-free token estimate.
    Roughly counts words + punctuation. Works well enough for a session budget.
    """
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

def azure_chat_stream(messages: List[Dict[str, str]], temperature: float = 0.2) -> Generator[str, None, None]:
    model = require_env(AZURE_CHAT_DEPLOYMENT, "AZURE_OPENAI_DEPLOYMENT")
    client = get_azure_client()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for event in stream:
        for choice in getattr(event, "choices", []):
            delta = getattr(choice, "delta", None)
            if delta and delta.content:
                yield delta.content

def generate_with_rag(
    user_text: str,
    system_prompt: Optional[str] = None,
    match_count: Optional[int] = None,
    mcp_client: Optional[MCPClientProtocol] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    if mcp_client:
        hits = mcp_client.retrieve_context(user_text, match_count=match_count)
    else:
        hits = retrieve_matches(user_text, match_count=match_count)
    system = system_prompt if system_prompt else get_system_prompt()
    ctx = format_context(hits)

    messages = [
        {"role": "system", "content": f"{system}\n\nCONTEXT:\n{ctx}"},
        {"role": "user", "content": user_text},
    ]

    response_text = ""
    for delta in azure_chat_stream(messages):
        response_text += delta
        yield ("delta", {"text": response_text})

    sources_block = build_sources_block(hits)
    final = (response_text or "").rstrip()
    if sources_block:
        final += "\n\n**Sources**\n" + sources_block
    yield ("final", {"text": final, "hits": hits})
