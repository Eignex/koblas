#!/usr/bin/env bash
# usage: compare.sh REPORT_DIR CANDIDATE REFERENCE [ARM]
# Prints "case reference_ns ratio" for every case both targets measured in one report directory, which is one
# machine's pass. ratio is candidate over reference, below 1 when the candidate is faster: the median per CPU model
# (the report's "model name"), combined over the models with equal weight. reference_ns is the median over every
# such pass. A lab job's fetched directory is a REPORT_DIR, whatever its case layout.
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

# One line per measured case and pass: case, reference time, candidate over reference, CPU model. Both times come
# from one report directory, which one machine wrote in one attempt.
pairs() {
  find "${sources[@]}" -name "$candidate.csv" -print0 | while IFS= read -r -d '' file; do
    local directory=${file%/*}
    [[ -f $directory/$reference.csv ]] || continue
    local cpu=
    [[ ! -f $directory/metadata.txt ]] || cpu=$(sed -n '/^model name=/{s///p;q;}' "$directory/metadata.txt")
    awk -F, -v cpu="${cpu:-unknown}" '
      FNR == 1 { for (i = 1; i <= NF; i++) column[$i] = i; next }
      $column["status"] != "ok" { next }
      FILENAME == ARGV[1] { reference[$column["case"]] = $column["median_ns"]; next }
      $column["case"] in reference {
        id = $column["case"]; gsub(/[ \t]+/, "_", cpu)
        print id, reference[id], $column["median_ns"] / reference[id], cpu
      }' "$directory/$reference.csv" "$file"
  done
}

# The ratio is each CPU model's median, combined over the models with equal weight, so a different mix of
# machines between jobs cannot move it; the reference time is the median over every pass.
pairs | awk '
  function median(list,   n, t, i, j, v) {
    n = split(substr(list, 2), t, " ")
    for (i = 2; i <= n; i++) { v = t[i]; for (j = i - 1; j >= 1 && t[j] > v; j--) t[j + 1] = t[j]; t[j + 1] = v }
    return (n % 2) ? t[(n + 1) / 2] : (t[n / 2] + t[n / 2 + 1]) / 2
  }
  { times[$1] = times[$1] " " $2; ratios[$1 SUBSEP $4] = ratios[$1 SUBSEP $4] " " $3 }
  END {
    for (key in ratios) { split(key, k, SUBSEP); logs[k[1]] += log(median(ratios[key])); models[k[1]]++ }
    for (id in times) printf "%s %.1f %.4f\n", id, median(times[id]), exp(logs[id] / models[id])
  }' | LC_ALL=C sort
