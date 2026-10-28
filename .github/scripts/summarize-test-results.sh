#!/usr/bin/env bash
# Keep recorded test time separate from compiler and simulator startup time.
set -euo pipefail
shopt -s nullglob
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT

for directory in "$@"; do
  reports=("$directory"/TEST-*.xml)
  ((${#reports[@]})) || continue
  : >"$temporary/cases"
  for report in "${reports[@]}"; do
    xmllint --nonet --format "$report" | awk '
      function attribute(name, value) {
        if (!match($0, "[[:space:]]" name "=\"[^\"]*\"")) return ""
        value = substr($0, RSTART, RLENGTH)
        sub(/^[^=]*="/, "", value)
        sub(/"$/, "", value)
        return value
      }
      /<testcase[ >]/ {
        printf "%.9f\t%s.%s\n", attribute("time"), attribute("classname"), attribute("name")
      }' >>"$temporary/cases"
  done
  printf '### %s\n\n' "$directory"
  awk -F '\t' '{ seconds += $1; count++ } END {
    printf "%d tests; %.3f seconds of recorded test time.\n\n", count, seconds
  }' "$temporary/cases"
  printf '| Slowest tests | Seconds |\n| --- | ---: |\n'
  sort -nr "$temporary/cases" | awk -F '\t' 'NR <= 10 {
    gsub(/[|]/, "\\\\|", $2)
    printf "| %s | %.3f |\n", $2, $1
  }'
  printf '\n'
done
