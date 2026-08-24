#!/usr/bin/env python3
"""Turn a kotlinx-benchmark JMH report into per-arm curves, for reading a dispatch crossover off.

    koblas-bench/tools/bench-curves.py koblas-bench/build/reports/benchmarks/level1/*/jvm.json

Emits CSV on stdout (benchmark,arm,size,ns,error) and a fit per benchmark and arm on stderr. The fit is
what makes a missing crossover legible: a gate only exists where two curves cross, so two arms with the
same slope have none at any threshold however far their intercepts differ.
"""
import json
import sys
from collections import defaultdict

# The parameter naming the size swept, and the one naming which side answered.
SIZE_KEYS = ("len", "n", "nrhs")
ARM_KEYS = ("backend",)


def rows(path):
    for entry in json.load(open(path)):
        params = entry.get("params", {})
        size = next((int(params[k]) for k in SIZE_KEYS if k in params), None)
        if size is None:
            continue
        arm = next((params[k] for k in ARM_KEYS if k in params), "-")
        metric = entry["primaryMetric"]
        yield (entry["benchmark"].rsplit(".", 1)[-1], arm, size,
               metric["score"], metric.get("scoreError") or 0.0)


def fit(points):
    """Least squares ns = intercept + slope * size."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(s for s, _ in points)
    sy = sum(v for _, v in points)
    sxx = sum(s * s for s, _ in points)
    sxy = sum(s * v for s, v in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    return (sy - slope * sx) / n, slope


def main(paths):
    data = defaultdict(list)
    print("benchmark,arm,size,ns,error")
    for path in paths:
        for bench, arm, size, ns, err in sorted(rows(path), key=lambda r: (r[0], r[1], r[2])):
            print(f"{bench},{arm},{size},{ns:.4f},{err:.4f}")
            data[(bench, arm)].append((size, ns))
    print("\nfit: ns = intercept + slope * size", file=sys.stderr)
    for (bench, arm), points in sorted(data.items()):
        parameters = fit(points)
        if parameters is None:
            continue
        intercept, slope = parameters
        widest = max(points)
        print(f"  {bench:<18} {arm:<10} intercept={intercept:9.2f} ns  slope={slope:8.4f} ns/unit  "
              f"at {widest[0]}: {widest[1] / widest[0]:.4f} ns/unit", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
