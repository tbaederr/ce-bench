#!/usr/bin/env python3
"""
Benchmark clang's constant interpreter against the default evaluator.

Runs each test file in tests/ through perf stat twice:
  1. Without -fexperimental-new-constant-interpreter (baseline)
  2. With -fexperimental-new-constant-interpreter (new interpreter)

Collects task-clock, instructions, cpu-cycles, and wall time.
"""

import argparse
import configparser
import html
import json
import re
import shlex
import subprocess
import urllib.request
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TESTS_DIR = SCRIPT_DIR / "tests"
SETTINGS_FILE = SCRIPT_DIR / "settings.ini"
RESULTS_DIR = SCRIPT_DIR / "perf-results"

DEFAULT_CLANG = "/usr/bin/clang"
PERF_REPETITIONS = 5

# Events we care about from perf stat output.
EVENTS_OF_INTEREST = [
    "task-clock:u",
    "instructions:u",
    "cpu-cycles:u",
    "page-faults:u",
    "branches:u",
    "branch-misses:u",
]

INTERP_FLAG = "-fexperimental-new-constant-interpreter"


def load_settings() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(SETTINGS_FILE)
    return config


def get_clang(config: configparser.ConfigParser) -> str:
    return config.get("tools", "clang", fallback=DEFAULT_CLANG)


def get_llvm_repo(config: configparser.ConfigParser) -> Path | None:
    val = config.get("tools", "llvm_repo", fallback="")
    if not val:
        return None
    return Path(val)


def read_extra_args(test_file: Path) -> list[str]:
    """Read extra clang args from the first line if it starts with '//'."""
    with open(test_file) as f:
        first_line = f.readline().strip()

    if not first_line.startswith("//"):
        return []

    return shlex.split(first_line[2:])


@dataclass
class PerfResult:
    """Parsed perf stat results for a single run."""
    events: dict[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0


def run_perf(cmd: list[str]) -> PerfResult:
    """Run `perf stat --json -r N -- <cmd>` and parse the JSON lines."""
    event_list = ",".join(EVENTS_OF_INTEREST)
    perf_cmd = [
        "perf", "stat",
        "--json",
        "-e", event_list,
        "-r", str(PERF_REPETITIONS),
        "--", *cmd,
    ]

    proc = subprocess.run(
        perf_cmd,
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        print(f"  FAILED: {' '.join(cmd)}", file=sys.stderr)
        print(f"  stderr: {proc.stderr[:500]}", file=sys.stderr)
        return PerfResult()

    result = PerfResult()

    for line in proc.stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = entry.get("event", "")
        value = entry.get("counter-value", "")
        if not event or value == "<not counted>":
            continue

        # Normalize perf event names to match EVENTS_OF_INTEREST.
        #   "cpu_core/instructions/u" -> "instructions:u"
        #   "task-clock:u"            -> "task-clock:u"
        #   "page-faults:u"           -> "page-faults:u"
        if "/" in event:
            parts = event.split("/")
            # cpu_core/instructions/u -> base="instructions", suffix="u"
            base = parts[1] if len(parts) >= 2 else parts[0]
            suffix = f":{parts[2]}" if len(parts) >= 3 else ""
            short = base + suffix
        else:
            short = event

        if short == "task-clock:u":
            result.wall_seconds = float(value) / 1000.0

        for name in EVENTS_OF_INTEREST:
            if short == name:
                # Prefer cpu_core over cpu_atom (first valid wins).
                if name not in result.events:
                    result.events[name] = float(value)
                break

    return result


def format_value(val: float) -> str:
    if val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"{val / 1_000:.1f}K"
    return f"{val:.2f}"


def delta_pct(baseline: float, new: float) -> float | None:
    if baseline == 0:
        return None
    return ((new - baseline) / baseline) * 100.0


def format_delta(baseline: float, new: float) -> str:
    pct = delta_pct(baseline, new)
    if pct is None:
        return "N/A"
    rounded = round(pct, 1)
    if rounded == 0.0:
        return "0.0%"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


@dataclass
class TestResult:
    name: str
    result: PerfResult
    failed: bool = False


def run_benchmarks(clang: str) -> list[TestResult]:
    test_files = sorted(TESTS_DIR.glob("*.cpp"))
    if not test_files:
        print("No .cpp files found in tests/", file=sys.stderr)
        sys.exit(1)

    print(f"Using clang: {clang}")
    print(f"Found {len(test_files)} tests, {PERF_REPETITIONS} repetitions each.\n")

    results = []
    for test in test_files:
        name = test.stem
        print(f"=== {name} {'=' * (60 - len(name))}")

        extra_args = read_extra_args(test)
        cmd = [clang, "-c", str(test), "-o", "/dev/null", "-std=c++20",
               "-fconstexpr-steps=0", INTERP_FLAG] + extra_args

        print(f"  Running ...", end="", flush=True)
        result = run_perf(cmd)
        print(" done.\n")

        failed = not result.events
        results.append(TestResult(name, result, failed))

        if failed:
            print("  SKIPPED (compilation failed)\n")
            continue

        print_terminal_table(result)

    return results


def print_terminal_table(result: PerfResult):
    print(f"  {'Event':<20} {'Value':>12}")
    print(f"  {'-' * 20} {'-' * 12}")

    for event in EVENTS_OF_INTEREST:
        val = result.events.get(event, 0)
        if val == 0:
            continue
        print(f"  {event:<20} {format_value(val):>12}")

    if result.wall_seconds:
        print(f"\n  Wall time: {result.wall_seconds:.3f}s")

    print()


# ── Persistence ─────────────────────────────────────────────────


@dataclass
class CommitInfo:
    hash: str
    subject: str


@dataclass
class RunRecord:
    """A full benchmark run, serializable to JSON."""
    commit: CommitInfo
    clang: str
    repetitions: int
    tests: list[TestResult]


def result_exists(commit_hash: str) -> bool:
    return (RESULTS_DIR / f"{commit_hash}.json").exists()


def save_results(record: RunRecord):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{record.commit.hash}.json"

    data = {
        "commit": record.commit.hash,
        "subject": record.commit.subject,
        "clang": record.clang,
        "repetitions": record.repetitions,
        "tests": [
            {
                "name": t.name,
                "failed": t.failed,
                "events": t.result.events,
                "wall_seconds": t.result.wall_seconds,
            }
            for t in record.tests
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    print(f"Results saved: {path}")


def load_all_results() -> list[RunRecord]:
    if not RESULTS_DIR.exists():
        return []

    records = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        tests = []
        for t in data.get("tests", []):
            tests.append(TestResult(
                name=t["name"],
                result=PerfResult(events=t.get("events", {}), wall_seconds=t.get("wall_seconds", 0)),
                failed=t.get("failed", False),
            ))

        commit = CommitInfo(
            hash=data.get("commit", path.stem),
            subject=data.get("subject", ""),
        )
        records.append(RunRecord(
            commit=commit,
            clang=data.get("clang", ""),
            repetitions=data.get("repetitions", 0),
            tests=tests,
        ))

    return records


def sort_by_git_order(llvm_repo: Path, records: list[RunRecord]) -> list[RunRecord]:
    """Sort records by git history order (oldest first)."""
    if not records:
        return records

    hashes = [r.commit.hash for r in records]
    proc = subprocess.run(
        ["git", "log", "--format=%H", "--no-walk", *hashes],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return records

    # git log --no-walk outputs newest first.
    ordered = proc.stdout.strip().splitlines()
    order_map = {h: i for i, h in enumerate(reversed(ordered))}
    by_hash = {r.commit.hash: r for r in records}

    return [by_hash[h] for h in reversed(ordered) if h in by_hash]


# ── LLVM repo queries ───────────────────────────────────────────

INITIAL_COMMIT_COUNT = 25
COMMIT_PREFIX = "[clang][bytecode]"
BYTECODE_PATH = "clang/lib/AST/ByteCode"


def _run_git_log(llvm_repo: Path, extra_args: list[str]) -> list[CommitInfo]:
    cmd = ["git", "log", "--format=%H %s"] + extra_args
    proc = subprocess.run(cmd, cwd=llvm_repo, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"git log failed: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    commits = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        hash_, subject = line.split(" ", 1)
        commits.append(CommitInfo(hash=hash_, subject=subject))

    return commits


def find_bytecode_commits(llvm_repo: Path, limit: int | None = None,
                          since: str | None = None) -> list[CommitInfo]:
    """Find commits by message prefix OR by touching BYTECODE_PATH.

    If `since` is set, only return commits after that hash.
    """
    revision = f"{since}..HEAD" if since else "--all"

    # Query 1: commits matching the message patterns.
    escaped = COMMIT_PREFIX.replace("[", "\\[").replace("]", "\\]")
    grep_args = [revision,
                 f"--grep=^{escaped}",
                 f'--grep=^Revert "{escaped}',
                 f'--grep=^Reapply "{escaped}']

    # Query 2: commits touching the bytecode directory.
    path_args = [revision, "--", BYTECODE_PATH]

    grep_commits = _run_git_log(llvm_repo, grep_args)
    path_commits = _run_git_log(llvm_repo, path_args)

    # Merge, deduplicate, preserve git log order (newest first).
    seen = set()
    merged = []
    for c in grep_commits + path_commits:
        if c.hash not in seen:
            seen.add(c.hash)
            merged.append(c)

    # Re-sort by git history order (newest first).
    if merged:
        hashes = [c.hash for c in merged]
        proc = subprocess.run(
            ["git", "log", "--format=%H", "--no-walk", *hashes],
            cwd=llvm_repo, capture_output=True, text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            order = proc.stdout.strip().splitlines()
            by_hash = {c.hash: c for c in merged}
            merged = [by_hash[h] for h in order if h in by_hash]

    if limit is not None:
        merged = merged[:limit]

    return merged


def find_newest_result(llvm_repo: Path) -> str:
    """Find the most recent commit (by git history) among existing results."""
    hashes = [p.stem for p in RESULTS_DIR.glob("*.json")]
    if not hashes:
        return ""

    # Ask git to sort them chronologically; the first one is the newest.
    proc = subprocess.run(
        ["git", "log", "--format=%H", "--no-walk", *hashes],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return hashes[0]

    return proc.stdout.strip().splitlines()[0]


def prune_stale_results(llvm_repo: Path):
    """Remove result files for commits no longer in the repo."""
    if not RESULTS_DIR.exists():
        return

    result_files = list(RESULTS_DIR.glob("*.json"))
    if not result_files:
        return

    hashes = [p.stem for p in result_files]

    # git cat-file --batch-check tells us which hashes exist.
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        input="\n".join(hashes),
        cwd=llvm_repo,
        capture_output=True, text=True,
    )

    for path, line in zip(result_files, proc.stdout.strip().splitlines()):
        if "missing" in line:
            print(f"Removing stale result: {path.name}")
            path.unlink()


def update_repo(llvm_repo: Path):
    print("Updating llvm repo ...", end="", flush=True)

    checkout = subprocess.run(
        ["git", "checkout", "main"],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if checkout.returncode != 0:
        print(" FAILED (checkout main)", file=sys.stderr)
        print(checkout.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    pull = subprocess.run(
        ["git", "pull", "origin", "main", "--rebase"],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if pull.returncode != 0:
        print(" FAILED (pull)", file=sys.stderr)
        print(pull.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    print(" done.")


def checkout_commit(llvm_repo: Path, commit_hash: str):
    proc = subprocess.run(
        ["git", "checkout", commit_hash],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"git checkout failed: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)


def build_clang(llvm_repo: Path):
    build_dir = llvm_repo / "build"
    print("  Building clang ...", end="", flush=True)
    proc = subprocess.run(
        ["ninja", "clang"],
        cwd=build_dir,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(" FAILED", file=sys.stderr)
        print(proc.stderr[-1000:], file=sys.stderr)
        sys.exit(1)
    print(" done.")


# ── PR check ────────────────────────────────────────────────────

GITHUB_PR_URL_RE = re.compile(
    r'https?://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)'
)


def parse_pr_url(url: str) -> tuple[str, int]:
    """Extract (owner/repo, pr_number) from a GitHub PR URL."""
    m = GITHUB_PR_URL_RE.match(url)
    if not m:
        print(f"Error: invalid PR URL: {url}", file=sys.stderr)
        sys.exit(1)
    return m.group(1), int(m.group(2))


def download_patch(repo_slug: str, pr_number: int) -> str:
    url = f"https://github.com/{repo_slug}/pull/{pr_number}.patch"
    print(f"Downloading patch for PR #{pr_number} ...", end="", flush=True)
    try:
        with urllib.request.urlopen(url) as resp:
            patch = resp.read().decode()
    except Exception as e:
        print(f" FAILED", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)
    print(" done.")
    return patch


def apply_patch(llvm_repo: Path, patch: str):
    print("  Applying patch ...", end="", flush=True)
    proc = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(" FAILED", file=sys.stderr)
        print(proc.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    print(" done.")


def revert_patch(llvm_repo: Path, patch: str):
    print("  Reverting patch ...", end="", flush=True)
    proc = subprocess.run(
        ["git", "apply", "-R", "-"],
        input=patch,
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(" FAILED (manual cleanup may be needed)", file=sys.stderr)
    else:
        print(" done.")


def get_head_commit(llvm_repo: Path) -> CommitInfo:
    proc = subprocess.run(
        ["git", "log", "-1", "--format=%H %s"],
        cwd=llvm_repo,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"git log failed: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    line = proc.stdout.strip()
    hash_, subject = line.split(" ", 1)
    return CommitInfo(hash=hash_, subject=subject)


def load_result(commit_hash: str) -> RunRecord:
    path = RESULTS_DIR / f"{commit_hash}.json"
    data = json.loads(path.read_text())
    tests = []
    for t in data.get("tests", []):
        tests.append(TestResult(
            name=t["name"],
            result=PerfResult(events=t.get("events", {}),
                              wall_seconds=t.get("wall_seconds", 0)),
            failed=t.get("failed", False),
        ))
    commit = CommitInfo(hash=data.get("commit", path.stem),
                        subject=data.get("subject", ""))
    return RunRecord(commit=commit, clang=data.get("clang", ""),
                     repetitions=data.get("repetitions", 0), tests=tests)


def print_check_results(baseline: RunRecord, pr_tests: list[TestResult],
                        pr_number: int):
    baseline_map = {t.name: t for t in baseline.tests}

    print(f"\n{'=' * 66}")
    print(f"  PR #{pr_number} vs main ({baseline.commit.hash[:12]})")
    print(f"{'=' * 66}\n")

    print(f"  {'Test':<20} {'Main':>14} {'PR':>14} {'Delta':>10}")
    print(f"  {'-' * 20} {'-' * 14} {'-' * 14} {'-' * 10}")

    for t in pr_tests:
        if t.failed:
            print(f"  {t.name:<20} {'':>14} {'FAILED':>14} {'':>10}")
            continue

        pr_instr = t.result.events.get("instructions:u", 0)
        base_t = baseline_map.get(t.name)
        if not base_t or base_t.failed:
            print(f"  {t.name:<20} {'':>14} {format_value(pr_instr):>14} {'':>10}")
            continue

        base_instr = base_t.result.events.get("instructions:u", 0)
        delta = format_delta(base_instr, pr_instr)
        print(f"  {t.name:<20} {format_value(base_instr):>14} {format_value(pr_instr):>14} {delta:>10}")

    print()


def run_check(llvm_repo: Path, pr_url: str):
    repo_slug, pr_number = parse_pr_url(pr_url)
    patch = download_patch(repo_slug, pr_number)

    update_repo(llvm_repo)
    main_head = get_head_commit(llvm_repo)

    # Benchmark main HEAD as baseline if needed.
    if not result_exists(main_head.hash):
        print(f"\nBenchmarking main ({main_head.hash[:12]}) as baseline ...")
        build_clang(llvm_repo)
        clang = built_clang_path(llvm_repo)
        results = run_benchmarks(clang)
        record = RunRecord(commit=main_head, clang=clang,
                           repetitions=PERF_REPETITIONS, tests=results)
        save_results(record)

    baseline = load_result(main_head.hash)

    # Apply patch, build, benchmark, then revert.
    apply_patch(llvm_repo, patch)
    try:
        build_clang(llvm_repo)
        clang = built_clang_path(llvm_repo)
        pr_results = run_benchmarks(clang)
    finally:
        revert_patch(llvm_repo, patch)

    print_check_results(baseline, pr_results, pr_number)


# ── HTML report ──────────────────────────────────────────────────

REPORT_FILE = SCRIPT_DIR / "report.html"


def delta_css_class(pct: float | None) -> str:
    if pct is None:
        return ""
    if round(pct, 1) == 0.0:
        return "unchanged"
    return "better" if pct < 0 else "worse"


def collect_test_names(records: list[RunRecord]) -> list[str]:
    """Gather unique test names that still have a file in tests/."""
    existing = {p.stem for p in TESTS_DIR.glob("*.cpp")}
    seen = set()
    names = []
    for r in records:
        for t in r.tests:
            if t.name not in seen and t.name in existing:
                seen.add(t.name)
                names.append(t.name)
    return names


def generate_html(records: list[RunRecord]):
    test_names = collect_test_names(records)

    # Header row: Commit | test1 | test2 | ...
    TESTS_BASE_URL = "https://github.com/tbaederr/ce-bench/tree/main/tests"

    header_cells = "<th>Commit</th>"
    for name in test_names:
        test_url = f"{TESTS_BASE_URL}/{name}.cpp"
        header_cells += f'<th class="num"><a href="{test_url}">{html.escape(name)}</a></th>'

    # Records are sorted oldest-first; build per-test instruction counts
    # so we can compute deltas between consecutive commits.
    # prev_instructions[test_name] = instruction count from the previous commit.
    prev_instructions: dict[str, float] = {}
    md_summaries: dict[str, str] = {}

    COPY_ICON = (
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="2">'
        '<rect x="9" y="9" width="13" height="13" rx="2"/>'
        '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>'
        '</svg>'
    )

    # Build rows oldest-first, then reverse for display (most recent on top).
    row_list = []
    for record in records:
        short_hash = record.commit.hash[:12]
        commit_url = f"https://github.com/llvm/llvm-project/commit/{record.commit.hash}"
        raw_subject = record.commit.subject
        subject = html.escape(raw_subject)
        subject = re.sub(r"`([^`]+)`", r"<code>\1</code>", subject)
        subject = re.sub(
            r"#(\d+)",
            r'<a href="https://github.com/llvm/llvm-project/pull/\1">#\1</a>',
            subject,
        )

        test_map = {t.name: t for t in record.tests}

        cells = ""
        all_zero = True
        has_delta = False
        md_test_rows = []
        for name in test_names:
            t = test_map.get(name)
            if not t or t.failed:
                cells += '<td class="num failed">—</td>'
                md_test_rows.append((name, "—", ""))
                continue

            cur = t.result.events.get("instructions:u", 0)
            prev = prev_instructions.get(name)
            prev_instructions[name] = cur

            abs_str = format_value(cur)
            if prev is None or prev == 0:
                cells += f'<td class="num">{abs_str}</td>'
                md_test_rows.append((name, abs_str, ""))
                continue

            has_delta = True
            pct = delta_pct(prev, cur)
            css = delta_css_class(pct)
            delta_str = format_delta(prev, cur)
            if round(pct, 1) != 0.0:
                all_zero = False
            cells += f'<td class="num">{abs_str} <span class="{css}">({delta_str})</span></td>'
            md_test_rows.append((name, abs_str, delta_str))

        # Markdown summary for clipboard.
        any_delta = any(d for _, _, d in md_test_rows)
        md_lines = []
        if any_delta:
            md_lines.append("| Test | Instructions | Delta |")
            md_lines.append("|------|-------------|-------|")
            for tname, tval, tdelta in md_test_rows:
                tlink = f"[{tname}]({TESTS_BASE_URL}/{tname}.cpp)"
                md_lines.append(f"| {tlink} | {tval} | {tdelta} |")
        else:
            md_lines.append("| Test | Instructions |")
            md_lines.append("|------|-------------|")
            for tname, tval, _ in md_test_rows:
                tlink = f"[{tname}]({TESTS_BASE_URL}/{tname}.cpp)"
                md_lines.append(f"| {tlink} | {tval} |")
        md_summaries[short_hash] = "\n".join(md_lines)

        commit_cell = (
            f'<td>'
            f'<a href="{commit_url}"><code>{html.escape(short_hash)}</code></a> '
            f'{subject} '
            f'<button class="copy-btn" data-hash="{short_hash}" '
            f'onclick="copyMd(this)" title="Copy markdown summary">'
            f'{COPY_ICON}</button>'
            f'</td>'
        )

        is_zero = all_zero and has_delta
        row_list.append((f"<tr>{commit_cell}{cells}</tr>\n", is_zero))

    # Group consecutive all-zero-delta rows and collapse groups > 1.
    # Rows are oldest-first; we reverse for display (newest on top).
    reversed_rows = list(reversed(row_list))
    rows = ""
    i = 0
    group_id = 0
    while i < len(reversed_rows):
        row_html, is_zero = reversed_rows[i]
        if not is_zero:
            rows += row_html
            i += 1
            continue

        # Collect consecutive zero-delta rows.
        group = []
        while i < len(reversed_rows) and reversed_rows[i][1]:
            group.append(reversed_rows[i][0])
            i += 1

        if len(group) == 1:
            rows += group[0]
            continue

        # Collapsible group.
        gid = f"zg{group_id}"
        group_id += 1
        toggle_row = (
            f'<tr class="toggle-row" onclick="toggleGroup(\'{gid}\')">'
            f'<td colspan="{1 + len(test_names)}" class="toggle-cell">'
            f'<span id="{gid}-arrow" class="arrow">&#9654;</span> '
            f'{len(group)} commits with no change</td></tr>\n'
        )
        rows += toggle_row
        for row_html in group:
            rows += row_html.replace("<tr>", f'<tr class="{gid} collapsed">', 1)

    md_json = json.dumps(md_summaries)

    page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bytecode Interpreter Benchmark</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; padding: 2.5rem;
    background: #f6f8fa; color: #1f2328;
    font-size: 0.85rem; line-height: 1.5;
  }}
  a {{ text-decoration: none; color: #0969da; }}
  a:hover {{ text-decoration: underline; }}
  h1 {{
    font-size: 1.3rem; font-weight: 600;
    margin: 0 0 0.4rem 0; color: #1f2328;
  }}
  .subtitle {{
    color: #656d76; font-size: 0.8rem; margin-bottom: 1.5rem;
  }}
  .table-wrap {{
    background: #fff; border: 1px solid #d0d7de; border-radius: 6px;
    overflow-x: auto;
  }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid #d0d7de;
    text-align: left; white-space: nowrap;
  }}
  th {{
    background: #f6f8fa; color: #656d76;
    font-weight: 600; font-size: 0.8rem;
    position: sticky; top: 0;
  }}
  tr:last-child td {{ border-bottom: none; }}
  td code {{
    background: #eff1f3; padding: 0.15rem 0.4rem;
    border-radius: 4px; font-size: 0.8em;
    font-family: ui-monospace, "SFMono-Regular", "SF Mono", Menlo, monospace;
  }}
  .num {{
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, "SFMono-Regular", "SF Mono", Menlo, monospace;
    font-size: 0.8rem;
  }}
  .better {{ color: #1a9f3b; font-weight: 600; }}
  .worse  {{ color: #cf222e; font-weight: 600; }}
  .failed {{ color: #8b949e; font-style: italic; }}
  .unchanged {{ color: #8b949e; }}
  .footer {{
    margin-top: 1.2rem; font-size: 0.75rem; color: #8b949e;
  }}
  .footer a {{ color: #8b949e; }}
  .footer a:hover {{ color: #656d76; }}
  .collapsed {{ display: none; }}
  .toggle-row {{ cursor: pointer; }}
  .toggle-row:hover {{ background: #f6f8fa; }}
  .toggle-cell {{ color: #656d76; font-size: 0.8rem; }}
  .arrow {{ font-size: 0.7rem; display: inline-block; transition: transform 0.15s; margin-right: 0.4rem; }}
  .arrow.open {{ transform: rotate(90deg); }}
  .copy-btn {{
    background: none; border: none;
    color: #656d76; cursor: pointer; padding: 2px 4px;
    margin-left: 0.4rem; vertical-align: middle; line-height: 1;
  }}
  .copy-btn:hover {{ background: #f6f8fa; color: #1f2328; }}
  .copy-btn.copied {{ color: #1a9f3b; }}
</style>
</head>
<body>
<h1>Bytecode Interpreter Benchmark</h1>
<p class="subtitle">instructions:u delta between consecutive commits &middot; {len(records)} commits</p>
<div class="table-wrap">
<table>
<thead>
  <tr>{header_cells}</tr>
</thead>
<tbody>
{rows}</tbody>
</table>
</div>
<p class="footer"><a href="https://github.com/tbaederr/ce-bench">Source Code</a></p>
<script>
var mdSummaries = {md_json};

function toggleGroup(gid) {{
  document.querySelectorAll("." + gid).forEach(function(r) {{
    r.classList.toggle("collapsed");
  }});
  document.getElementById(gid + "-arrow").classList.toggle("open");
}}

function copyMd(btn) {{
  var md = mdSummaries[btn.dataset.hash];
  if (!md) return;
  navigator.clipboard.writeText(md).then(function() {{
    btn.classList.add("copied");
    setTimeout(function() {{ btn.classList.remove("copied"); }}, 1500);
  }});
}}
</script>
</body>
</html>
"""

    REPORT_FILE.write_text(page)
    print(f"HTML report: file://{REPORT_FILE.resolve()}")


# ── Entry point ──────────────────────────────────────────────────

def built_clang_path(llvm_repo: Path) -> str:
    return str(llvm_repo / "build" / "bin" / "clang")


def run_for_commit(commit: CommitInfo, llvm_repo: Path,
                    current: int = 0, total: int = 0):
    """Checkout a commit, build clang, run benchmarks, save results."""
    short = commit.hash[:12]
    progress = f" ({current} of {total})" if total else ""
    print(f"\n{'#' * 66}")
    print(f"# {short} — {commit.subject}{progress}")
    print(f"{'#' * 66}\n")

    checkout_commit(llvm_repo, commit.hash)
    build_clang(llvm_repo)

    clang = built_clang_path(llvm_repo)
    results = run_benchmarks(clang)
    record = RunRecord(commit=commit, clang=clang,
                       repetitions=PERF_REPETITIONS, tests=results)
    save_results(record)


def main():
    parser = argparse.ArgumentParser(description="Constexpr interpreter benchmarks")
    parser.add_argument("--update", action="store_true",
                        help="Update the llvm repo and run benchmarks for new commits")
    parser.add_argument("--check", metavar="PR_URL",
                        help="Benchmark a GitHub PR against main")
    args = parser.parse_args()

    config = load_settings()
    llvm_repo = get_llvm_repo(config)

    if not llvm_repo:
        print("Error: llvm_repo not set in settings.ini", file=sys.stderr)
        sys.exit(1)

    if not llvm_repo.is_dir():
        print(f"Error: {llvm_repo} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.update:
        update_repo(llvm_repo)
        prune_stale_results(llvm_repo)

        first_run = not RESULTS_DIR.exists() or not any(RESULTS_DIR.glob("*.json"))

        if first_run:
            new_commits = find_bytecode_commits(llvm_repo, limit=INITIAL_COMMIT_COUNT)
        else:
            newest_hash = find_newest_result(llvm_repo)
            new_commits = find_bytecode_commits(llvm_repo, since=newest_hash)

        if new_commits:
            print(f"Benchmarking {len(new_commits)} new commit(s):")
            for c in new_commits:
                print(f"  {c.hash[:12]} {c.subject}")
            print()
            ordered = list(reversed(new_commits))
            for i, commit in enumerate(ordered, 1):
                run_for_commit(commit, llvm_repo, current=i, total=len(ordered))
        else:
            print("No new commits to benchmark.")

    if args.check:
        run_check(llvm_repo, args.check)

    all_records = load_all_results()
    all_records = sort_by_git_order(llvm_repo, all_records)
    generate_html(all_records)


if __name__ == "__main__":
    main()
