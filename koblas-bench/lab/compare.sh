#!/usr/bin/env bash
# usage: compare.sh REPORT_DIR CANDIDATE REFERENCE [ARM]
# Prints "case reference_ns ratio" for every case both targets measured, where each time is the median of the
# target's median_ns over every pass and cell report under REPORT_DIR, and ratio is candidate over reference:
# below 1 the candidate is faster. A lab job's fetched directory is a REPORT_DIR, whatever its case layout.
#
# A fetched job lists its cases in lab-cases.jsonl. Only cases of ARM count, and only from rows in which every
# arm finished, so the arms of an A/B job are compared over the same machines. A job with one arm needs no ARM.
set -euo pipefail
[[ $# == 3 || $# == 4 ]] || { echo "usage: compare.sh REPORT_DIR CANDIDATE REFERENCE [ARM]" >&2; exit 2; }
directory=$1 candidate=$2 reference=$3 arm=${4:-}

sources=("$directory")
index=$directory/lab-cases.jsonl
if [[ -f $index ]]; then
  # One line per case: index, arm, row (problem, seed and repeat) and status.
  cases=$(awk '{
    match($0, /"index":[0-9]+/); i = substr($0, RSTART + 8, RLENGTH - 8)
    match($0, /"arm":"[^"]*"/); a = substr($0, RSTART + 7, RLENGTH - 8)
    match($0, /"problem":"[^"]*"/); p = substr($0, RSTART, RLENGTH)
    match($0, /"seed":[^,]*/); s = substr($0, RSTART, RLENGTH)
    match($0, /"repeat":[0-9]+/); r = substr($0, RSTART, RLENGTH)
    match($0, /"status":"[A-Z]+"/); t = substr($0, RSTART + 10, RLENGTH - 11)
    print i, a, p s r, t
  }' "$index")
  arms=$(awk '{ print $2 }' <<<"$cases" | sort -u)
  if [[ -z $arm ]]; then
    [[ $(wc -l <<<"$arms") == 1 ]] || { echo "compare.sh: name one of the job's arms:" $arms >&2; exit 2; }
    arm=$arms
  fi
  grep -Fxq -- "$arm" <<<"$arms" || { echo "compare.sh: no arm $arm; the job has:" $arms >&2; exit 2; }
  sources=()
  while read -r case; do sources+=("$directory/cases/$case"); done < <(awk -v arm="$arm" '
    { row[NR] = $3; armOf[NR] = $2; id[NR] = $1; if ($4 != "DONE") broken[$3] = 1 }
    END { for (n = 1; n <= NR; n++) if (armOf[n] == arm && !(row[n] in broken)) print id[n] }' <<<"$cases")
  ((${#sources[@]})) || { echo "compare.sh: arm $arm has no row in which every arm finished" >&2; exit 1; }
elif [[ -n $arm ]]; then
  echo "compare.sh: ARM needs a fetched lab job with lab-cases.jsonl" >&2; exit 2
fi

medians() {
  find "${sources[@]}" -name "$1.csv" -print0 | xargs -0 -r awk -F, '
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
