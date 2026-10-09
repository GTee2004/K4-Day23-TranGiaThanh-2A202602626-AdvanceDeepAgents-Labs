"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REFERENCE_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REFERENCE_LINE = re.compile(r"^\[(\d+)\]")
_CITATION_GROUP = re.compile(
    r"\[((?:\d+(?:\s*[-\u2013]\s*\d+)?)"
    r"(?:\s*,\s*\d+(?:\s*[-\u2013]\s*\d+)?)*)\](?!\()"
)
_CODE = re.compile(r"(```.*?```|~~~.*?~~~|`+[^`\n]*`+)", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>]+")


def _citation_numbers(group):
    """Expand a citation group such as ``1, 3-5`` into integers."""
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[-\u2013]\s*(\d+)", part)
        if not span:
            numbers.append(int(part))
            continue

        start, end = int(span.group(1)), int(span.group(2))
        if start <= end and end - start <= 200:
            numbers.extend(range(start, end + 1))
        else:
            # Treat a malformed or unreasonably large range as its two
            # explicit endpoints rather than allocating an enormous list.
            numbers.extend((start, end))
    return numbers


def _citations_outside_code(text):
    """Return citation numbers, excluding fenced/inline code and links."""
    cited = set()
    for index, segment in enumerate(_CODE.split(text)):
        if index % 2:  # Captured separators are code segments.
            continue
        for match in _CITATION_GROUP.finditer(segment):
            cited.update(_citation_numbers(match.group(1)))
    return cited


def _urls(line):
    """Extract HTTP(S) URLs from one reference line."""
    return [match.group(0) for match in _URL.finditer(line)]


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []

    if not isinstance(sources, list):
        return ["sources.json must be a JSON list"]
    if not sources:
        return ["no sources in sources.json"]

    source_numbers = set()
    source_urls = {}
    seen_urls = {}

    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} is not an object")
            continue

        number = source.get("n")
        url = source.get("url")

        if type(number) is not int:  # bool is an int subclass but not a valid n.
            problems.append(f"source entry {index} has non-integer n={number!r}")
        elif number in source_numbers:
            problems.append(f"source [{number}] appears more than once in sources.json")
        else:
            source_numbers.add(number)
            source_urls[number] = url

        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            label = f"[{number}]" if type(number) is int else f"entry {index}"
            problems.append(f"source {label} has invalid url={url!r}")
        elif url in seen_urls:
            problems.append(
                f"source URL duplicated by [{seen_urls[url]}] and [{number}]: {url}"
            )
        else:
            seen_urls[url] = number

    headings = list(_REFERENCE_HEADING.finditer(report_text))
    if not headings:
        problems.append("report is missing the '## References' heading")
        body = report_text
        references = ""
    else:
        heading = headings[0]
        body = report_text[:heading.start()]
        references = report_text[heading.end():]

    cited = _citations_outside_code(body)
    for number in sorted(cited - source_numbers):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(source_numbers - cited):
        problems.append(f"source [{number}] never cited")

    reference_lines = {}
    for line in references.splitlines():
        match = _REFERENCE_LINE.match(line)
        if not match:
            continue
        number = int(match.group(1))
        reference_lines.setdefault(number, []).append(line)

    for number in sorted(reference_lines):
        lines = reference_lines[number]
        if number not in source_numbers:
            problems.append(f"reference [{number}] missing from sources.json")
        if len(lines) > 1:
            problems.append(f"reference [{number}] appears more than once")

    for number in sorted(source_numbers):
        lines = reference_lines.get(number, [])
        if not lines:
            problems.append(f"source [{number}] has no reference line")
            continue

        # Validate every duplicate as well as reporting the duplicate itself.
        for line in lines:
            urls = _urls(line)
            if len(urls) != 1:
                problems.append(
                    f"reference [{number}] must contain exactly one URL (found {len(urls)})"
                )
            elif urls[0] != source_urls.get(number):
                problems.append(
                    f"reference [{number}] URL does not match sources.json: {urls[0]}"
                )

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
