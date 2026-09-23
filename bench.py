#!/usr/bin/env python3
"""
Benchmark clang's constant interpreter against the default evaluator.

Runs each test file in tests/ through perf stat twice:
  1. Without -fexperimental-new-constant-interpreter (baseline)
  2. With -fexperimental-new-constant-interpreter (new interpreter)

Collects task-clock, instructions, cpu-cycles, and wall time.
"""

import configparser
import html
import json
import re
import shlex
import subprocess
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
    print(f"Found {len(test_files)} test(s), {PERF_REPETITIONS} repetitions each.\n")

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

INITIAL_COMMIT_COUNT = 3
COMMIT_PREFIX = "[clang][bytecode]"


def find_bytecode_commits(llvm_repo: Path, limit: int | None = None,
                          since: str | None = None) -> list[CommitInfo]:
    """Find commits with subjects starting with COMMIT_PREFIX.

    If `since` is set, only return commits after that hash.
    """
    escaped = COMMIT_PREFIX.replace("[", "\\[").replace("]", "\\]")
    revision = f"{since}..HEAD" if since else "--all"
    cmd = ["git", "log", "--oneline", "--format=%H %s", revision,
           f"--grep=^{escaped}"]
    if limit is not None:
        cmd.append(f"-{limit}")

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


# ── HTML report ──────────────────────────────────────────────────

REPORT_FILE = SCRIPT_DIR / "report.html"


def delta_css_class(pct: float | None) -> str:
    if pct is None or round(pct, 1) == 0.0:
        return ""
    return "better" if pct < 0 else "worse"


def collect_test_names(records: list[RunRecord]) -> list[str]:
    """Gather all unique test names across runs, in stable order."""
    seen = set()
    names = []
    for r in records:
        for t in r.tests:
            if t.name not in seen:
                seen.add(t.name)
                names.append(t.name)
    return names


def generate_html(records: list[RunRecord]):
    test_names = collect_test_names(records)

    # Header row: Commit | test1 | test2 | ...
    header_cells = "<th>Commit</th>"
    for name in test_names:
        header_cells += f"<th>{html.escape(name)}</th>"

    # Records are sorted oldest-first; build per-test instruction counts
    # so we can compute deltas between consecutive commits.
    # prev_instructions[test_name] = instruction count from the previous commit.
    prev_instructions: dict[str, float] = {}

    # Build rows oldest-first, then reverse for display (most recent on top).
    row_list = []
    for record in records:
        short_hash = record.commit.hash[:12]
        commit_url = f"https://github.com/llvm/llvm-project/commit/{record.commit.hash}"
        subject = html.escape(record.commit.subject)
        subject = re.sub(
            r"#(\d+)",
            r'<a href="https://github.com/llvm/llvm-project/pull/\1">#\1</a>',
            subject,
        )
        commit_cell = (
            f'<td><a href="{commit_url}"><code>{html.escape(short_hash)}</code></a> '
            f'{subject}</td>'
        )

        test_map = {t.name: t for t in record.tests}

        cells = ""
        for name in test_names:
            t = test_map.get(name)
            if not t or t.failed:
                cells += '<td class="num failed">—</td>'
                continue

            cur = t.result.events.get("instructions:u", 0)
            prev = prev_instructions.get(name)
            prev_instructions[name] = cur

            abs_str = format_value(cur)
            if prev is None or prev == 0:
                cells += f'<td class="num">{abs_str}</td>'
                continue

            pct = delta_pct(prev, cur)
            css = delta_css_class(pct)
            delta_str = format_delta(prev, cur)
            cells += f'<td class="num">{abs_str} <span class="{css}">({delta_str})</span></td>'

        row_list.append(f"<tr>{commit_cell}{cells}</tr>\n")

    rows = "".join(reversed(row_list))

    page = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Constexpr Interpreter Benchmark</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #222; font-size: 0.85rem; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 0.4rem 0.8rem; border: 1px solid #ddd; text-align: left; white-space: nowrap; }}
  th {{ background: #f0f0f0; }}
  td code {{ background: #e8e8e8; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.85em; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .better {{ color: #1a7f37; font-weight: 600; }}
  .worse  {{ color: #cf222e; font-weight: 600; }}
  .failed {{ color: #888; font-style: italic; }}
</style>
</head>
<body>
<h1>Constexpr Interpreter Benchmark — Instructions Delta</h1>
<table>
<thead>
  <tr>{header_cells}</tr>
</thead>
<tbody>
{rows}</tbody>
</table>
</body>
</html>
"""

    REPORT_FILE.write_text(page)
    print(f"HTML report: {REPORT_FILE}")


# ── Entry point ──────────────────────────────────────────────────

def built_clang_path(llvm_repo: Path) -> str:
    return str(llvm_repo / "build" / "bin" / "clang")


def run_for_commit(commit: CommitInfo, llvm_repo: Path):
    """Checkout a commit, build clang, run benchmarks, save results."""
    short = commit.hash[:12]
    print(f"\n{'#' * 66}")
    print(f"# {short} — {commit.subject}")
    print(f"{'#' * 66}\n")

    checkout_commit(llvm_repo, commit.hash)
    build_clang(llvm_repo)

    clang = built_clang_path(llvm_repo)
    results = run_benchmarks(clang)
    record = RunRecord(commit=commit, clang=clang,
                       repetitions=PERF_REPETITIONS, tests=results)
    save_results(record)


def main():
    config = load_settings()
    llvm_repo = get_llvm_repo(config)

    if not llvm_repo:
        print("Error: llvm_repo not set in settings.ini", file=sys.stderr)
        sys.exit(1)

    if not llvm_repo.is_dir():
        print(f"Error: {llvm_repo} is not a directory", file=sys.stderr)
        sys.exit(1)

    update_repo(llvm_repo)

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
        # Oldest first so the HTML report shows progression.
        for commit in reversed(new_commits):
            run_for_commit(commit, llvm_repo)
    else:
        print("No new commits to benchmark.")

    all_records = load_all_results()
    all_records = sort_by_git_order(llvm_repo, all_records)
    generate_html(all_records)


if __name__ == "__main__":
    main()
