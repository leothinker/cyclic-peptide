"""Choose between the readings of an activity cell -- offline.

``phase7_activity_dump.py`` records the four readings of every
activity cell (the whole value crop, the printed line, the Chinese
detector and the English detector).  Which of them is used is a
*policy*.  Evaluating a policy means reading every cell again, which
takes half an hour, so the readings are dumped once and the policies
are compared here, against the archived extraction, in seconds.

The policies are deliberately few and each one states the rule it
follows, so that the winner is a rule and not a coincidence.

Usage::

    .venv\\Scripts\\python.exe experiments/phase7_activity_policy.py
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO / "src", REPO):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cpd.ocr.activity import parse_number  # noqa: E402

ACTIVITY_FIELDS = ("kd_nm", "rt_min", "ms_mz", "lcms_method", "ms_polarity")
SOURCES = ("crop", "line", "ch", "en")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", default="data/audit/phase7/activity_dump.csv"
    )
    parser.add_argument(
        "--archive",
        default=(
            "data/processed/archive/"
            "compounds_combined_production_route.csv"
        ),
    )
    return parser.parse_args(argv)


def _normalise(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", text).upper()


def _comparable(field: str, value: str | None) -> str:
    if value is None:
        return ""
    if field in ("kd_nm", "rt_min", "ms_mz"):
        number = parse_number(value)
        return "" if number is None else repr(number)
    return " ".join(value.split())


def _readings(row: dict[str, str]) -> list[tuple[str, str, int]]:
    """The three readings of one cell as ``(source, text, boxes)``."""
    result = []
    for source in SOURCES:
        text = row.get(f"{source}_text") or ""
        if not text.strip():
            continue
        boxes = int(row.get(f"{source}_boxes") or 0)
        result.append((source, text, boxes))
    return result


def policy_line_first(row: dict[str, str]) -> str:
    """The printed line wins.

    It is the only reading that cannot lose a character to the
    detector's segmentation.
    """
    for source in ("line", "crop", "ch", "en"):
        text = row.get(f"{source}_text") or ""
        if text.strip():
            return text
    return ""


def policy_ch_first(row: dict[str, str]) -> str:
    """The detector readings come first, the line is the fallback."""
    for source in ("ch", "en", "line", "crop"):
        text = row.get(f"{source}_text") or ""
        if text.strip():
            return text
    return ""


def policy_en_first(row: dict[str, str]) -> str:
    """The English detector comes first."""
    for source in ("en", "ch", "line", "crop"):
        text = row.get(f"{source}_text") or ""
        if text.strip():
            return text
    return ""


def policy_crop_first(row: dict[str, str]) -> str:
    """The whole value crop wins.

    The crop reading treats the cell as one printed value, which is
    what a retention time or an m/z is.  It cannot pick up the
    residue of the column rules, and it cannot lose the decimal
    point.
    """
    for source in ("crop", "line", "ch", "en"):
        text = row.get(f"{source}_text") or ""
        if text.strip():
            return text
    return ""


def policy_crop_for_numbers(row: dict[str, str]) -> str:
    """A numeric cell uses the crop reading, a text cell the line.

    The two readings fail in opposite ways.  The crop reading merges
    the two printed lines of a gradient cell into ``42-100-``, so it
    is wrong for ``lcms_method``.  The line reading loses the decimal
    point of ``1.53`` and reaches past the value into the column
    rules, so it is wrong for the numbers and for the polarity.

    The split between the two is therefore made by the *kind* of the
    field, which the profile states, and not by a rule that has to
    look at the value.
    """
    numeric = row.get("numeric") == "1"
    order = ("crop", "line", "ch", "en") if numeric else (
        "line", "crop", "ch", "en"
    )
    for source in order:
        text = row.get(f"{source}_text") or ""
        if text.strip():
            return text
    return ""


def policy_agreeing_group(row: dict[str, str]) -> str:
    """The largest group of readings that agree wins.

    Every reading is compared with every other one, and the value
    that the most readings agree on is taken.  A reading that had to
    be split into several detector boxes cannot outvote a reading
    that was recognised as one token.
    """
    readings = _readings(row)
    if not readings:
        return ""

    numeric = row.get("numeric") == "1"

    def key(text: str) -> object:
        if numeric:
            number = parse_number(text)
            if number is not None:
                return ("number", number)
        return ("text", _normalise(text))

    groups: dict[object, list[tuple[str, str, int]]] = {}
    for reading in readings:
        groups.setdefault(key(reading[1]), []).append(reading)

    eligible = [
        group for group in groups.values()
        if any(reading[2] <= 1 for reading in group)
    ]
    if not eligible:
        eligible = list(groups.values())

    def rank(group: list[tuple[str, str, int]]) -> tuple:
        has_crop = any(reading[0] == "crop" for reading in group)
        has_line = any(reading[0] == "line" for reading in group)
        return (
            len(group),
            1 if has_crop else 0,
            1 if has_line else 0,
            -min(reading[2] for reading in group),
        )

    best = max(eligible, key=rank)
    if len(best) > 1:
        for source in ("ch", "en"):
            for reading in best:
                if reading[0] == source:
                    return reading[1]
    order = {"crop": 0, "line": 1, "ch": 2, "en": 3}
    return min(best, key=lambda r: (r[2], order.get(r[0], 4)))[1]


def policy_current(row: dict[str, str]) -> str:
    """The rule that is in ``cpd.ocr.activity`` right now."""
    readings = _readings(row)
    if not readings:
        return ""

    numeric = row.get("numeric") == "1"

    def key(text: str) -> object:
        if numeric:
            number = parse_number(text)
            if number is not None:
                return ("number", number)
        return ("text", _normalise(text))

    groups: dict[object, list[tuple[str, str, int]]] = {}
    for reading in readings:
        groups.setdefault(key(reading[1]), []).append(reading)
    eligible = [
        group for group in groups.values()
        if any(reading[2] <= 1 for reading in group)
    ]
    if not eligible:
        eligible = list(groups.values())
    best = max(
        eligible,
        key=lambda group: (
            len(group),
            -min(reading[2] for reading in group),
        ),
    )
    if len(best) > 1:
        for source in ("ch", "en"):
            for reading in best:
                if reading[0] == source:
                    return reading[1]
    order = {"crop": 0, "line": 1, "ch": 2, "en": 3}
    return min(best, key=lambda r: (r[2], order.get(r[0], 3)))[1]


def policy_number_aware(row: dict[str, str]) -> str:
    """A numeric cell is read by a reading that yields a number.

    A reading that does not parse as a number cannot be the value of a
    numeric cell, so it is only used when nothing else parses.  Among
    the readings that do parse, the ones that agree win; if they do
    not agree, the longest reading is taken, because a reading that
    lost a character is shorter than the value that was printed.
    """
    readings = _readings(row)
    if not readings:
        return ""

    if row.get("numeric") == "1":
        numbers = [
            reading for reading in readings
            if parse_number(reading[1]) is not None
        ]
        if numbers:
            readings = numbers
        groups: dict[object, list[tuple[str, str, int]]] = {}
        for reading in readings:
            groups.setdefault(parse_number(reading[1]), []).append(reading)
        best = max(
            groups.values(),
            key=lambda group: (
                len(group),
                max(len(reading[1]) for reading in group),
                -min(reading[2] for reading in group),
            ),
        )
        return max(best, key=lambda r: (len(r[1]), -r[2]))[1]

    groups = {}
    for reading in readings:
        groups.setdefault(_normalise(reading[1]), []).append(reading)
    best = max(
        groups.values(),
        key=lambda group: (
            len(group),
            max(len(reading[1]) for reading in group),
        ),
    )
    return max(best, key=lambda r: len(r[1]))[1]


def policy_line_tiebreak(row: dict[str, str]) -> str:
    """Agreement first; on a disagreement the printed line wins.

    This is the rule the architecture document states.  The readings
    are grouped by the value they stand for, and a group that holds a
    reading which needed several detector boxes is set aside: such a
    reading has already lost information.  The largest agreeing group
    wins, and if two groups are equally large the one that holds the
    printed-line reading wins, because that reading is the only one
    that cannot have lost a character to the detector's segmentation.
    """
    readings = _readings(row)
    if not readings:
        return ""

    numeric = row.get("numeric") == "1"

    def key(text: str) -> object:
        if numeric:
            number = parse_number(text)
            if number is not None:
                return ("number", number)
        return ("text", _normalise(text))

    groups: dict[object, list[tuple[str, str, int]]] = {}
    for reading in readings:
        groups.setdefault(key(reading[1]), []).append(reading)
    eligible = [
        group for group in groups.values()
        if any(reading[2] <= 1 for reading in group)
    ]
    if not eligible:
        eligible = list(groups.values())

    def rank(group: list[tuple[str, str, int]]) -> tuple:
        has_line = any(reading[0] == "line" for reading in group)
        return (
            len(group),
            1 if has_line else 0,
            -min(reading[2] for reading in group),
        )

    best = max(eligible, key=rank)
    if len(best) > 1:
        # The readings agree; the detector's text keeps the format of
        # the column.
        for source in ("ch", "en"):
            for reading in best:
                if reading[0] == source:
                    return reading[1]
    order = {"crop": 0, "line": 1, "ch": 2, "en": 3}
    return min(best, key=lambda r: (r[2], order.get(r[0], 3)))[1]


POLICIES = {
    "line_first": policy_line_first,
    "ch_first": policy_ch_first,
    "en_first": policy_en_first,
    "crop_first": policy_crop_first,
    "crop_numbers": policy_crop_for_numbers,
    "agreeing_group": policy_agreeing_group,
    "current": policy_current,
    "number_aware": policy_number_aware,
    "line_tiebreak": policy_line_tiebreak,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dump_path = REPO / args.dump
    if not dump_path.exists():
        print(f"dump not found: {dump_path}")
        print("run experiments/phase7_activity_dump.py first")
        return 2
    with dump_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (REPO / args.archive).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        archive = {r["cmpd_id"]: r for r in csv.DictReader(handle)}

    print(f"{len(rows)} cells, {len(archive)} archived compounds\n")
    header = f"{'policy':<16}"
    for field in ACTIVITY_FIELDS:
        header += f"{field[:9]:>11}"
    header += f"{'TOTAL':>9}"
    print(header)
    print("-" * len(header))

    for name, policy in POLICIES.items():
        counts = {field: 0 for field in ACTIVITY_FIELDS}
        total = 0
        for row in rows:
            field = row["field"]
            if field not in counts:
                continue
            reference = archive.get(row["cmpd_id"])
            if reference is None:
                continue
            want = _comparable(field, reference.get(field))
            got = _comparable(field, policy(row))
            if want == got:
                counts[field] += 1
                total += 1
        line = f"{name:<16}"
        for field in ACTIVITY_FIELDS:
            line += f"{counts[field]:>11}"
        line += f"{total:>9}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
