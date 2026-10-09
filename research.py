"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from agents import (
    FINALIZER_PATH,
    REPORT_PATH,
    SOURCES_PATH,
    VALIDATOR_PATH,
    WORKDIR,
    build_lead_agent,
)
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def _progress(message):
    print(f"[progress] {message}", flush=True)


def _safe_error_message(exc):
    """Keep credentials and URL query strings out of terminal errors."""
    message = str(exc)
    for name, value in os.environ.items():
        if value and len(value) >= 4 and name.upper().endswith(("KEY", "TOKEN", "PASSWORD", "SECRET")):
            message = message.replace(value, "<redacted>")
    return re.sub(
        r"https?://[^\s'\"<>]+",
        lambda match: match.group(0).split("?", 1)[0] + "?<redacted>"
        if "?" in match.group(0) else match.group(0),
        message,
    )


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", str(topic).strip().lower(), flags=re.UNICODE)
    slug = slug.strip("-_")[:60].rstrip("-_")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        "Complete the full deep-research workflow for the topic below. Follow the "
        "workspace contract and every planning, delegation, source-diversity, "
        "finalization, validation, and spot-check step in your system instructions.\n\n"
        f"Research topic: {str(topic).strip()}"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0

    def value(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    for message in messages or []:
        calls = value(message, "tool_calls") or []
        if not calls:
            additional = value(message, "additional_kwargs", {}) or {}
            if isinstance(additional, dict):
                calls = additional.get("tool_calls", []) or []

        for call in calls:
            name = value(call, "name")
            if not name:
                function = value(call, "function", {}) or {}
                name = value(function, "name")
            if name:
                tool_calls[str(name)] += 1

        usage = value(message, "usage_metadata") or {}
        if usage:
            input_tokens += int(value(usage, "input_tokens", value(usage, "input", 0)) or 0)
            output_tokens += int(value(usage, "output_tokens", value(usage, "output", 0)) or 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("the agent did not create a non-empty report.md")
    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError("the agent did not create a non-empty sources.json")

    try:
        report_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("report.md is not valid UTF-8") from exc
    try:
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("sources.json is not valid UTF-8 JSON") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON array")
    if not all(isinstance(source, dict) for source in sources):
        raise RuntimeError("every sources.json entry must be an object")

    summary = summarize(messages, elapsed, model_name)
    source_families = sorted(
        {
            source.get("source")
            for source in sources
            if isinstance(source.get("source"), str) and source.get("source")
        }
    )
    metadata = {
        "topic": str(topic).strip(),
        **summary,
        "n_sources": len(sources),
        "source_families": source_families,
    }
    meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"

    reports_dir = Path(reports_dir)
    slug = slugify(topic)
    outputs = {
        reports_dir / f"{slug}.md": report_bytes,
        reports_dir / f"{slug}.sources.json": sources_bytes,
        reports_dir / f"{slug}.meta.json": meta_bytes,
    }

    # All validation above happens before the first host-side write. Temporary
    # files also prevent an interrupted write from leaving a truncated report.
    reports_dir.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{time.time_ns()}"
    staged = []
    try:
        for path, content in outputs.items():
            temporary = path.with_name(f".{path.name}.{nonce}.tmp")
            temporary.write_bytes(content)
            staged.append((temporary, path))
        for temporary, path in staged:
            temporary.replace(path)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()

    return reports_dir / f"{slug}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic).strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    _progress(f"research start: {topic}")
    try:
        _progress("model start")
        model = make_model()
        _progress("model ready")
        start = time.monotonic()
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or type(model).__name__
        )

        _progress("sandbox start")
        with open_sandbox() as backend:
            _progress("sandbox ready")
            _progress("preparing workspace")
            created = backend.execute(
                f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report"
            )
            if created.exit_code != 0:
                raise RuntimeError(f"cannot create sandbox workspace: {created.output}")

            _progress("uploading validator and finalizer")
            upload(
                backend,
                {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                },
            )
            seeded = backend.execute(
                f"test -s {VALIDATOR_PATH} && test -s {FINALIZER_PATH}"
            )
            if seeded.exit_code != 0:
                raise RuntimeError("citation scripts were not uploaded to the sandbox")

            _progress("building agent")
            agent = build_lead_agent(backend, model)
            _progress("agent running")
            agent_started = time.monotonic()
            stop_heartbeat = threading.Event()

            def heartbeat():
                while not stop_heartbeat.wait(30):
                    _progress(f"still running: {int(time.monotonic() - agent_started)}s")

            heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
            heartbeat_thread.start()
            try:
                result = agent.invoke(
                    {
                        "messages": [
                            {"role": "user", "content": build_prompt(topic)}
                        ]
                    },
                    config={"recursion_limit": 1000},
                )
            finally:
                stop_heartbeat.set()
                heartbeat_thread.join(timeout=1)
            _progress("agent finished")
            messages = result.get("messages", []) if isinstance(result, dict) else []
            elapsed = time.monotonic() - start
            _progress("downloading outputs")
            report_path = save_outputs(
                backend, topic, messages, elapsed, model_name
            )
            _progress(f"report saved: {report_path}")
    except Exception as exc:
        error = _safe_error_message(exc)
        _progress(f"failed: {type(exc).__name__}: {error}")
        print(f"FAILED: {type(exc).__name__}: {error}", file=sys.stderr)
        return 1

    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
