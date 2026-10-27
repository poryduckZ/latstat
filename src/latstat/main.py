import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from typing import IO

from stats import Dist, Summary, parse, summarise


def run(
    args: Sequence[str],
    stdin: IO[str] = sys.stdin,
    stdout: IO[str] = sys.stdout,
    stderr: IO[str] = sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(
        prog="latstat",
        description="Summarise LLM request logs.",
        add_help=False,
    )

    parser.add_argument("--by", default="", help='group results by field (e.g. "backend")')
    parser.add_argument("--json", dest="as_json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("file", nargs="?", default=None, help="JSON Lines file path (defaults to stdin)")

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit:
        return 2

    if parsed_args.file is None:
        parsed = parse(stdin)
    else:
        try:
            with open(parsed_args.file, encoding="utf-8") as f:
                parsed = parse(f)
        except OSError as err:
            print(f"latstat: {err}", file=stderr)
            return 1

    for n in parsed.bad_lines:
        print(f"latstat: skipping malformed line {n}", file=stderr)

    if not parsed.records:
        print("latstat: no records", file=stderr)
        return 1

    try:
        summaries = summarise(parsed.records, parsed_args.by)
    except ValueError as err:
        print(f"latstat: {err}", file=stderr)
        return 2

    if parsed_args.as_json:
        summary_dicts = [dataclasses.asdict(s) for s in summaries]
        json.dump(summary_dicts, stdout, indent=2)
        stdout.write("\n")
        return 0

    print_table(stdout, summaries)
    return 0


def print_table(out: IO[str], summaries: list[Summary]) -> None:
    """Format and print a right-aligned columnar table, equivalent to Go's tabwriter."""
    headers = [
        "group", "reqs", "err%",
        "ttft_p50", "ttft_p99",
        "tpot_p50", "tpot_p99",
        "e2e_p50", "e2e_p99",
        "tok/s",
    ]

    rows: list[list[str]] = [headers]
    for s in summaries:
        rows.append([
            s.group,
            str(s.requests),
            f"{s.error_rate * 100:.1f}",
            ms(s.ttft, 50),
            ms(s.ttft, 99),
            ms(s.tpot, 50),
            ms(s.tpot, 99),
            ms(s.e2e, 50),
            ms(s.e2e, 99),
            f"{s.tokens_per_sec:.1f}",
        ])

    widths = [max(map(len, col)) for col in zip(*rows)]
    for row in rows:
        print("".join(cell.rjust(w + 2) for cell, w in zip(row, widths)), file=out)



def ms(d: Dist | None, p: int) -> str:
    if not d:
        return "-"

    if p == 50:
        return f"{d.p50:.1f}"

    return f"{d.p99:.1f}"


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
