#!/usr/bin/env bash
# Reject empty or failing test runs, including tests removed by a shrinker.
set -euo pipefail
shopt -s nullglob

(($#)) || { echo 'usage: verify-test-results.sh DIRECTORY...' >&2; exit 2; }
for directory in "$@"; do
  reports=("$directory"/TEST-*.xml)
  ((${#reports[@]})) || { echo "No test reports in $directory" >&2; exit 1; }
  for report in "${reports[@]}"; do
    xmllint --nonet --noout "$report"
    valid=$(xmllint --nonet --xpath \
      'boolean(/*[@tests > 0 and (not(@failures) or @failures = 0) and (not(@errors) or @errors = 0) and (not(@skipped) or @skipped < @tests)])' "$report")
    [[ $valid == true ]] || { echo "Empty or failing test report: $report" >&2; exit 1; }
    tests=$(xmllint --nonet --xpath 'string(/*/@tests)' "$report")
    echo "Verified $tests executed tests: $report"
  done
done
