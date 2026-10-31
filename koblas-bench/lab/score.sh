#!/usr/bin/env bash
# usage: score.sh VENDOR COMPARISON...
# Verdicts per routine and size band from compare.sh or median.sh output ("case reference_ns ratio").
#
# Bands: Level 1 small below 4096 elements, mid to 16,384, large from 65,000; Level 2 by matrix elements (a
# symmetric or triangular order counts its full square), small below 16,384, mid below 65,000, large above.
# A small cell wins only when every row is at most 1.00; a mid or large cell wins when the geometric mean of
# its ratios is at most 1.00. With VENDOR accelerate, a losing cell whose reference runs faster than FLOOR ns
# per element on every row (default 0.09) is reported as coprocessor rather than lose.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: score.sh VENDOR COMPARISON..." >&2; exit 2; }
vendor=$1
shift
cat "$@" | awk -v vendor="$vendor" -v floor="${FLOOR:-0.09}" '
  { line[NR] = $0; if ($1 ~ /\+timing=arithmetic/) { b = $1; sub(/\+timing=arithmetic/, "", b); arith[b] = 1 } }
  END {
    for (r = 1; r <= NR; r++) {
      split(line[r], f, " "); id = f[1]; ns = f[2]; ratio = f[3]
      # A case timed without the reset replaces its reset-timed twin.
      if (id !~ /\+timing=arithmetic/ && (id in arith)) continue
      split(id, p, "+"); op = p[1]
      triangle = op ~ /^(trsv|trmv|syr|syr2|symv)$/
      level2 = triangle || op ~ /^(gemv|ger)$/
      if (p[2] ~ /x/) { split(p[2], q, "x"); square = q[1] * q[2] } else square = level2 ? p[2] * p[2] : p[2] + 0
      elements = triangle ? p[2] * (p[2] + 1) / 2 : square
      if (level2) band = square >= 65000 ? "large" : (square >= 16384 ? "mid" : "small")
      else band = square >= 65000 ? "large" : (square >= 4096 ? "mid" : "small")
      k = op SUBSEP band
      n[k]++; lg[k] += log(ratio); if (ratio > mx[k]) mx[k] = ratio
      if (ns / elements < floor) fast[k]++
    }
    for (k in n) {
      split(k, q, SUBSEP)
      g = exp(lg[k] / n[k])
      verdict = (q[2] == "small" ? mx[k] <= 1.0 : g <= 1.0) ? "win" : "lose"
      if (vendor == "accelerate" && verdict == "lose" && fast[k] == n[k]) verdict = "coprocessor"
      printf "%-6s %-5s %-11s geomean %.2f  worst %.2f  rows %d\n", q[1], q[2], verdict, g, mx[k], n[k]
    }
  }' | sort -k1,1 -k2,2r
