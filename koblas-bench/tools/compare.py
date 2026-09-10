#!/usr/bin/env python3
import csv
import statistics
import sys

REQUIRED = {"case", "implementation", "workload_version", "fixture_version", "ns_per_op", "status", "comparison_kind", "timing_mode", "threads", "warmups", "target_ns"}
MATCH_FIELDS = ("workload_version", "fixture_version", "timing_mode", "threads", "comparison_kind", "warmups", "target_ns")

def load(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise SystemExit(f"{path}: incompatible or empty benchmark CSV")
    grouped = {}
    for row in rows:
        key = (row["case"], *(row[field] for field in MATCH_FIELDS))
        grouped.setdefault(key, []).append(row)
    return grouped

def main(paths):
    require_compatible = False
    if paths and paths[0] == "--require-compatible":
        require_compatible = True
        paths = paths[1:]
    if len(paths) < 2:
        raise SystemExit("usage: compare.py [--require-compatible] BASE.csv CANDIDATE.csv [CANDIDATE.csv ...]")
    base = load(paths[0])
    incompatible = 0
    for path in paths[1:]:
        candidate = load(path)
        print(f"candidate={path}")
        print("case,base_median_ns,candidate_median_ns,candidate_min_ns,candidate_max_ns,base_over_candidate,comparison_kind")
        for key in sorted(base.keys() & candidate.keys()):
            left = [float(row["ns_per_op"]) for row in base[key] if row["status"] == "ok"]
            right = [float(row["ns_per_op"]) for row in candidate[key] if row["status"] == "ok"]
            if not left or not right:
                continue
            kind = candidate[key][0]["comparison_kind"]
            a, b = statistics.median(left), statistics.median(right)
            print(f"{key[0]},{a:.9g},{b:.9g},{min(right):.9g},{max(right):.9g},{a / b:.6g},{kind}")
        base_ok = {key[0] for key, rows in base.items() if any(row["status"] == "ok" for row in rows)}
        candidate_ok = {key[0] for key, rows in candidate.items() if any(row["status"] == "ok" for row in rows)}
        joined = {key[0] for key in base.keys() & candidate.keys() if any(row["status"] == "ok" for row in base[key]) and any(row["status"] == "ok" for row in candidate[key])}
        for case in sorted((base_ok & candidate_ok) - joined):
            left = sorted({"/".join(key[1:]) for key in base if key[0] == case})
            right = sorted({"/".join(key[1:]) for key in candidate if key[0] == case})
            print(f"incompatible case={case} base={left} candidate={right}", file=sys.stderr)
            incompatible += 1
    if require_compatible and incompatible:
        raise SystemExit(f"{incompatible} supported case pairs have incompatible metadata")

if __name__ == "__main__":
    main(sys.argv[1:])
