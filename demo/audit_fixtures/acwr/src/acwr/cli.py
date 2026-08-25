"""Command-line interface: point it at a CSV of daily loads."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .core import AcwrPoint, Zone, ewma_acwr, monotony, rolling_acwr, strain

SPARK = "▁▂▃▄▅▆▇█"

# Zone -> (ANSI colour, glyph). Colour is dropped when not writing to a TTY.
STYLE = {
    Zone.UNDERLOADED: ("\033[38;5;110m", "·"),
    Zone.OPTIMAL: ("\033[38;5;114m", "●"),
    Zone.CAUTION: ("\033[38;5;179m", "▲"),
    Zone.HIGH_RISK: ("\033[38;5;174m", "■"),
    Zone.UNDEFINED: ("\033[38;5;243m", " "),
}
RESET = "\033[0m"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi == lo:
        return SPARK[0] * len(values)
    return "".join(
        SPARK[min(len(SPARK) - 1, int((v - lo) / (hi - lo) * len(SPARK)))]
        for v in values
    )


def read_csv(path: Path, column: str) -> tuple[list[str], list[float]]:
    """Read labels and loads from a CSV with a header row."""
    labels: list[str] = []
    loads: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: no header row")
        if column not in reader.fieldnames:
            raise SystemExit(
                f"{path}: no column named {column!r}. "
                f"Found: {', '.join(reader.fieldnames)}"
            )
        label_field = reader.fieldnames[0]
        for row_no, row in enumerate(reader, start=2):
            raw = (row.get(column) or "").strip()
            if raw == "":
                raw = "0"
            try:
                loads.append(float(raw))
            except ValueError:
                raise SystemExit(f"{path}:{row_no}: {raw!r} is not a number")
            labels.append((row.get(label_field) or str(row_no - 1)).strip())
    if not loads:
        raise SystemExit(f"{path}: no data rows")
    return labels, loads


def render(points: list[AcwrPoint], labels: list[str], colour: bool, tail: int) -> str:
    rows = list(zip(labels, points))[-tail:]
    width = max((len(label) for label, _ in rows), default=4)
    out = [f"{'day'.ljust(width)}  {'load':>7}  {'acute':>7}  {'chronic':>7}  ratio"]
    out.append("─" * (width + 40))
    for label, p in rows:
        tint, glyph = STYLE[p.zone]
        ratio = "    —" if p.ratio is None else f"{p.ratio:5.2f}"
        line = (
            f"{label.ljust(width)}  {p.load:7.1f}  {p.acute:7.1f}  "
            f"{p.chronic:7.1f}  {ratio} {glyph} {p.zone.value}"
        )
        out.append(f"{tint}{line}{RESET}" if colour else line)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="acwr",
        description="Acute:chronic workload ratio from a CSV of daily training loads.",
    )
    parser.add_argument("csv", type=Path, help="CSV file with a header row")
    parser.add_argument(
        "-c", "--column", default="load", help="load column name (default: load)"
    )
    parser.add_argument("-a", "--acute", type=int, default=7, help="acute days")
    parser.add_argument("-C", "--chronic", type=int, default=28, help="chronic days")
    parser.add_argument(
        "-m",
        "--method",
        choices=("rolling", "ewma"),
        default="rolling",
        help="ratio method (default: rolling)",
    )
    parser.add_argument(
        "-n", "--tail", type=int, default=14, help="rows to show (default: 14)"
    )
    parser.add_argument("--no-colour", action="store_true", help="disable colour")
    args = parser.parse_args(argv)

    labels, loads = read_csv(args.csv, args.column)

    compute = rolling_acwr if args.method == "rolling" else ewma_acwr
    try:
        points = compute(loads, args.acute, args.chronic)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    colour = not args.no_colour and sys.stdout.isatty()

    print()
    print(f"  {len(loads)} days · {args.method} · {args.acute}:{args.chronic}")
    print(f"  load  {sparkline(loads)}")
    ratios = [p.ratio for p in points if p.ratio is not None]
    if ratios:
        print(f"  acwr  {sparkline(ratios)}")
    print()
    print(render(points, labels, colour, args.tail))
    print()

    latest = points[-1]
    if latest.ratio is None:
        need = args.chronic - len(loads)
        print(
            f"  Ratio undefined — {need} more day(s) needed to fill the "
            f"{args.chronic}-day chronic window."
            if need > 0
            else "  Ratio undefined — no chronic load recorded."
        )
    else:
        print(f"  Latest ACWR {latest.ratio:.2f} ({latest.zone.value})")

    recent = loads[-7:]
    m = monotony(recent)
    if m is not None:
        print(f"  7-day monotony {m:.2f} · strain {strain(recent):,.0f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
