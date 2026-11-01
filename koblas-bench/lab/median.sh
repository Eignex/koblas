#!/usr/bin/env bash
# usage: median.sh COMPARISON...
# Combines compare.sh outputs from repeated runs into one, each case's reference time and ratio the median over
# the runs that measured it. It does not weigh CPU models, so use it only for runs on one model; to combine jobs
# whose machines differ, pool their cases in one directory and run compare.sh once.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: median.sh COMPARISON..." >&2; exit 2; }
cat "$@" | awk '
  { times[$1] = times[$1] " " $2; ratios[$1] = ratios[$1] " " $3 }
  function median(list,   n, t, i, j, v) {
    n = split(substr(list, 2), t, " ")
    for (i = 2; i <= n; i++) { v = t[i]; for (j = i - 1; j >= 1 && t[j] > v; j--) t[j + 1] = t[j]; t[j + 1] = v }
    return (n % 2) ? t[(n + 1) / 2] : (t[n / 2] + t[n / 2 + 1]) / 2
  }
  END { for (id in ratios) printf "%s %.1f %.4f\n", id, median(times[id]), median(ratios[id]) }' | LC_ALL=C sort
