"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

import httpx
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_INTERVAL = 3.0
_ARXIV_LOCK = threading.Lock()
_last_arxiv_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if base < 0 or cap < 0:
        raise ValueError("base and cap must be non-negative")

    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise

            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                try:
                    delay = min(cap, max(0.0, float(retry_after)))
                except (TypeError, ValueError):
                    delay = min(cap, base * (2**attempt) + random.uniform(0, base))
            else:
                delay = min(cap, base * (2**attempt) + random.uniform(0, base))
            time.sleep(delay)


def _retry_after(response):
    """Return a numeric Retry-After value when the server supplied one."""
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _request(method, url, **kwargs):
    """Make one HTTP request and turn retryable statuses into RetryableError."""
    response = httpx.request(method, url, timeout=30.0, follow_redirects=True, **kwargs)
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(
            f"HTTP {response.status_code} for {response.url}",
            retry_after=_retry_after(response),
        )
    response.raise_for_status()
    return response


def _clean(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit is not None else text


def _clamp(value, low, high):
    return max(low, min(high, int(value)))


def _json_or_no_results(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


def _error(exc, secret=""):
    """Format a tool error and redact Exa credentials from exception URLs."""
    message = str(exc)
    if secret:
        message = message.replace(secret, "<redacted>")
    message = re.sub(r"(?i)(exaApiKey=)[^&\s'\"]+", r"\1<redacted>", message)
    return f"ERROR: {type(exc).__name__}: {message}"


def _paper_record(item, *, prefer_ai_summary=False):
    """Normalize one Hugging Face API item; return None when it has no id."""
    if not isinstance(item, dict):
        return None
    paper = item.get("paper")
    if not isinstance(paper, dict):
        return None
    paper_id = _clean(paper.get("id"))
    if not paper_id:
        return None

    def field(name, default=None):
        value = paper.get(name)
        return item.get(name, default) if value is None else value

    if prefer_ai_summary:
        summary = field("ai_summary") or field("summary")
    else:
        summary = field("summary")
    published = _clean(field("publishedAt"))[:10]
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": published,
        "title": _clean(field("title")),
        "summary": _clean(summary, 600),
        "upvotes": field("upvotes", 0) or 0,
        "github": field("githubRepo", "") or "",
        "stars": field("githubStars", 0) or 0,
    }


def _score(record):
    try:
        return float(record.get("upvotes", 0))
    except (TypeError, ValueError):
        return 0.0


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    try:
        terms = re.findall(r"[^\W_]+(?:-[^\W_]+)*", str(query), flags=re.UNICODE)
        terms = [term for term in terms if term.lower() not in {"all", "and", "or", "not"}]
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _clamp(max_results, 1, 30),
            "start": 0,
        }

        def fetch():
            global _last_arxiv_call
            with _ARXIV_LOCK:
                wait = _ARXIV_INTERVAL - (time.monotonic() - _last_arxiv_call)
                if wait > 0:
                    time.sleep(wait)
                _last_arxiv_call = time.monotonic()
                return _request("GET", ARXIV_URL, params=params)

        response = with_retry(fetch, attempts=8, base=2.0, cap=60.0)
        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("atom:entry", namespace):
            raw_id = _clean(entry.findtext("atom:id", "", namespace))
            paper_id = raw_id.split("/abs/", 1)[-1].split("?", 1)[0]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            if not paper_id:
                continue
            published = _clean(entry.findtext("atom:published", "", namespace))[:10]
            records.append(
                {
                    "id": paper_id,
                    "url": f"https://arxiv.org/abs/{paper_id}",
                    "published": published,
                    "title": _clean(entry.findtext("atom:title", "", namespace)),
                    "summary": _clean(entry.findtext("atom:summary", "", namespace), 600),
                }
            )
        return _json_or_no_results(records)
    except Exception as exc:  # Tools must return errors, never raise them.
        return _error(exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": _clamp(limit, 1, 100)}
        if date:
            params["date"] = str(date)
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Hugging Face daily_papers returned a non-list response")
        records = [record for item in payload if (record := _paper_record(item))]
        needle = _clean(keyword).casefold()
        if needle:
            records = [
                record
                for record in records
                if needle in f"{record['title']} {record['summary']}".casefold()
            ]
        records.sort(key=_score, reverse=True)
        return _json_or_no_results(records)
    except Exception as exc:  # Tools must return errors, never raise them.
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        query = _clean(query)
        if not query:
            return "NO RESULTS"
        params = {"q": query, "limit": _clamp(limit, 1, 50)}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Hugging Face papers search returned a non-list response")
        records = [
            record
            for item in payload
            if (record := _paper_record(item, prefer_ai_summary=True))
        ]
        return _json_or_no_results(records)
    except Exception as exc:  # Tools must return errors, never raise them.
        return _error(exc)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    query = _clean(query)
    if not query:
        return "NO RESULTS"
    arguments = {
        "query": query,
        "objective": _clean(objective) or f"Find reliable, relevant sources about {query}",
        "numResults": _clamp(num_results, 1, 10),
    }
    return _call_exa("web_search_exa", arguments)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    url = _clean(url)
    if not url:
        return "NO RESULTS"
    result = _call_exa("web_fetch_exa", {"urls": [url]})
    return result if result.startswith(("ERROR:", "NO RESULTS")) else result[:12000]


def _exa_rate_limited(result, text):
    """Detect the HTTP-200 rate-limit form described by the lab guide."""
    meta = result.get("_meta")
    meta_text = json.dumps(meta, ensure_ascii=False).casefold() if meta is not None else ""
    markers = ("rate limit", "rate_limit", "ratelimit", "too many requests")
    if any(marker in meta_text for marker in markers):
        return True
    return bool(result.get("isError")) and any(marker in text.casefold() for marker in markers)


def _parse_mcp_response(response):
    """Parse either SSE `data:` events or a plain JSON MCP response."""
    messages = []
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data and data != "[DONE]":
            messages.append(json.loads(data))
    if messages:
        return messages[-1]
    return response.json()


def _call_exa(tool_name, arguments):
    """Call one Exa MCP tool and return its text without exposing credentials."""
    api_key = os.getenv("EXA_API_KEY", "").strip()
    params = {"exaApiKey": api_key} if api_key else None
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def call_once():
        response = _request(
            "POST", EXA_URL, params=params, headers=headers, json=payload
        )
        message = _parse_mcp_response(response)
        if not isinstance(message, dict):
            raise ValueError("Exa MCP returned an invalid JSON-RPC response")
        if "error" in message:
            error = message["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise RuntimeError(f"Exa JSON-RPC error: {detail}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise ValueError("Exa MCP response has no result object")
        texts = [
            part.get("text", "")
            for part in result.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        text = "\n".join(value for value in texts if value).strip()
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa MCP rate limited the request", retry_after=20)
        return text

    try:
        text = with_retry(call_once, attempts=6, base=2.0, cap=60.0)
        return text if text else "NO RESULTS"
    except Exception as exc:  # Tools must return errors, never raise them.
        return _error(exc, api_key)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
