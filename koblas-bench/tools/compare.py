#!/usr/bin/env python3
import csv
import statistics
import sys

REQUIRED = {"case", "implementation", "workload_version", "fixture_version", "ns_per_op", "status", "comparison_kind", "timing_mode", "threads"}

def load(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows or not REQUIRED.issubset(rows[0]):
        raise SystemExit(f"{path}: incompatible or empty benchmark CSV")
    grouped = {}
    for row in rows:
        key = (row["case"], row["workload_version"], row["fixture_version"], row["timing_mode"], row["threads"], row["comparison_kind"])
        grouped.setdefault(key, []).append(row)
    return grouped

def main(paths):
    if len(paths) < 2:
        raise SystemExit("usage: compare.py BASE.csv CANDIDATE.csv [CANDIDATE.csv ...]")
    base = load(paths[0])
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

if __name__ == "__main__":
    main(sys.argv[1:])
