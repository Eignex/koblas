#!/usr/bin/env bash
# usage: compare.sh REPORT_DIR CANDIDATE REFERENCE
# Prints "case reference_ns ratio" for every case both targets measured, where each time is the median of the
# target's median_ns over every pass and cell report under REPORT_DIR, and ratio is candidate over reference:
# below 1 the candidate is faster. A lab job's fetched directory is a REPORT_DIR, whatever its case layout.
set -euo pipefail
[[ $# == 3 ]] || { echo "usage: compare.sh REPORT_DIR CANDIDATE REFERENCE" >&2; exit 2; }
directory=$1 candidate=$2 reference=$3

medians() {
  find "$directory" -name "$1.csv" -print0 | xargs -0 -r awk -F, '
    FNR == 1 { for (i = 1; i <= NF; i++) column[$i] = i; next }
    $column["status"] == "ok" { print $column["case"], $column["median_ns"] }' |
    awk '{ times[$1] = times[$1] " " $2 }
    END {
      for (id in times) {
        n = split(substr(times[id], 2), t, " ")
        for (i = 2; i <= n; i++) { v = t[i]; for (j = i - 1; j >= 1 && t[j] > v; j--) t[j + 1] = t[j]; t[j + 1] = v }
        print id, (n % 2) ? t[(n + 1) / 2] : (t[n / 2] + t[n / 2 + 1]) / 2
      }
    }' | LC_ALL=C sort
}

LC_ALL=C join <(medians "$reference") <(medians "$candidate") | awk '{ printf "%s %.1f %.4f\n", $1, $2, $3 / $2 }'
