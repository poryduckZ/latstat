import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    id: str
    backend: str
    start_ms: float
    first_token_ms: float
    end_ms: float
    output_tokens: int
    status: str

    @property
    def ttft(self) -> float:
        """TTFT is time to first token: how long the user waits before anything appears."""
        return self.first_token_ms - self.start_ms

    @property
    def e2e(self) -> float:
        """E2E is the end-to-end latency for the whole request."""
        return self.end_ms - self.start_ms

    @property
    def tpot(self) -> float | None:
        """Time per output token after the first one; None if only one token."""
        if self.output_tokens < 2:
            return None
        return (self.end_ms - self.first_token_ms) / (self.output_tokens - 1)

@dataclass(frozen=True)
class ParseResult:
    records: list[Record]
    bad_lines: list[int]


def parse(r: Iterable[str]) -> ParseResult:
    """Parse read JSON lines. Blank lines are ignored and malformed lines are recorded (not fatal)"""
    records: list[Record] = []
    bad_lines: list[int] = []

    for line_num, line in enumerate(r, start=1):
        if not line.strip():
            continue

        try:
            records.append(Record(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            bad_lines.append(line_num)

    return ParseResult(records=records, bad_lines=bad_lines)

def percentile(values: Sequence[float], p: float) -> float:
    """
    Calculate a percentile using linear interpolation.
    values must be non-empty and sorted in ascending order.
    p must be in [0, 100]
    """
    length = len(values)

    if length == 1:
        return values[0]

    rank = p / 100 * (length - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    frac = rank - low
    return values[low] + (values[high] - values[low]) * frac


@dataclass(frozen=True)
class Dist:
    p50: float
    p90: float
    p99: float
    mean: float

def _summarise_dist(values: Iterable[float]) -> Dist | None:
    s = sorted(values)

    if not s:
        return None

    return Dist(
            p50=percentile(s, 50), 
            p90=percentile(s, 90), 
            p99=percentile(s, 99), 
            mean=sum(s) / len(s),
    )


@dataclass(frozen=True)
class Summary:
    group: str
    requests: int
    errors: int
    error_rate: float
    ttft: Dist | None
    tpot: Dist | None
    e2e: Dist | None
    output_tokens: int
    tokens_per_sec: float

def summarise(records: Iterable[Record], group_by: str) -> list[Summary]:
    if group_by and group_by != "backend":
        raise ValueError(f"unsupported group-by field {group_by!r} (want `backend` or none)")

    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        key = "all"
        if group_by == "backend":
            key = r.backend
            if key == "":
                key = "(none)"

        groups[key].append(r)

    result: list[Summary] = []

    for name in sorted(groups):
        recs = groups[name]
        errors = 0
        output_tokens = 0

        ttft_values: list[float] = []
        tpot_values: list[float] = []
        e2e_values: list[float] = []

        min_start = math.inf
        max_end = -math.inf

        for r in recs:
            if r.status != "ok":
                errors += 1
                continue

            ttft_values.append(r.ttft)
            e2e_values.append(r.e2e)

            if (tpot := r.tpot) is not None:
                tpot_values.append(tpot)

            output_tokens += r.output_tokens
            min_start = min(min_start, r.start_ms)
            max_end = max(max_end, r.end_ms)

        span = max_end - min_start
        tokens_per_sec = (
                output_tokens / (span / 1000)
                if span > 0
                else 0.0
        )

        result.append(
                Summary(
                    group=name,
                    requests=len(recs),
                    errors=errors,
                    error_rate=errors / len(recs),
                    ttft=_summarise_dist(ttft_values),
                    tpot=_summarise_dist(tpot_values),
                    e2e=_summarise_dist(e2e_values),
                    output_tokens=output_tokens,
                    tokens_per_sec=tokens_per_sec,
            )
        )

    return result
    