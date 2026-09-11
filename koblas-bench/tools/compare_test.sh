#!/usr/bin/env bash
set -euo pipefail

tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
temporary=$(mktemp -d)
trap 'rm -r "$temporary"' EXIT
base="$temporary/base.csv"
candidate="$temporary/candidate.csv"

fail() { echo "$*" >&2; exit 1; }

header() {
  echo 'run,id,implementation,pass,unit,source_commit,dirty,runtime,threads,warmups,target_ns,harness,warmup_target_ns,forks'
  echo 'case,id,run_id,case,status,comparison_kind,timing_mode,actual_kernel'
  echo 'sample,case_id,fork,sample,operations,elapsed_ns,ns_per_op'
}

row_id=0
row() {
  row_id=$((row_id + 1))
  gawk -v id="$row_id" '
  function csv(value) {
    if (value ~ /[",\r\n]/) { gsub(/"/, "\"\"", value); return "\"" value "\"" }
    return value
  }
  function record(type, fields, prefix,    names, count, i) {
    count = split(fields, names)
    printf "%s,%s", type, prefix
    for (i = 1; i <= count; i++) printf ",%s", csv(values[names[i]])
    print ""
  }
  BEGIN {
    run = "implementation pass unit source_commit dirty runtime threads warmups target_ns harness warmup_target_ns forks"
    case_fields = "case status comparison_kind timing_mode actual_kernel"
    sample = "fork sample operations elapsed_ns ns_per_op"
    count = split(run " " case_fields " " sample, fields)
    for (i = 1; i <= count; i++) values[fields[i]] = "1"
    values["timing_mode"] = "prepacked-compute"
    values["comparison_kind"] = "direct"
    values["case"] = "gemm-block+15x7x31+uniform+packed=4x4+timing=prepacked-compute"
    values["implementation"] = "jvm-c"
    values["actual_kernel"] = "c-tile"
    values["status"] = "ok"
    values["ns_per_op"] = "2"
    for (i = 1; i < ARGC; i++) {
      delimiter = index(ARGV[i], "=")
      values[substr(ARGV[i], 1, delimiter - 1)] = substr(ARGV[i], delimiter + 1)
    }
    record("run", run, id)
    record("case", case_fields, id "," id)
    if (values["status"] == "ok") record("sample", sample, id)
    exit
  }' -- "$@"
}

report() {
  local path=$1
  shift
  row_id=0
  { header; row "$@"; } >"$path"
}

compare() {
  local mode=$1
  shift
  "$tools/compare.sh" --require-compatible --mode "$mode" "$base" "$candidate" "$@" >"$temporary/output.csv" 2>"$temporary/error.txt"
}

accept() { compare "$@" || fail "comparison rejected: $(cat "$temporary/error.txt")"; }
reject() { if compare "$@"; then fail "incompatible comparison accepted"; fi; }
lines() { [[ $(wc -l <"$temporary/output.csv") -eq $1 ]] || fail "unexpected comparison pair count"; }

# Logical mode retains each layout pair instead of pooling distinct physical strategies.
report "$base"
report "$candidate"
row case=gemm-block+15x7x31+uniform+packed=8x4+timing=prepacked-compute >>"$candidate"
accept logical
lines 3
report "$candidate" case=gemm-block+15x7x31+uniform+packed=8x4+timing=prepacked-compute
reject fixed

# Mathematical identity survives renaming the actual kernel symbol.
report "$candidate" actual_kernel=new-symbol
accept fixed
lines 2

# Semantic and physical boundaries must match even when logical identity matches.
for field in timing_mode threads warmups target_ns; do
  report "$candidate" "$field=different"
  reject fixed
done
for timing in raw-tile packing-only layout-only vendor-arithmetic; do
  report "$base" "timing_mode=$timing"
  report "$candidate" "timing_mode=$timing" case=gemm-block+15x7x31+uniform+packed=8x4+timing=prepacked-compute
  reject logical
done

# Option order does not change identity; matrix flags and fixtures do.
report "$base"
report "$candidate" case=gemm-block+15x7x31+uniform+timing=prepacked-compute+packed=4x4
accept fixed
for value in 'gemm-block+15x7x31+other+packed=4x4+timing=prepacked-compute' \
  'gemm-block+15x7x31+uniform+packed=4x4+transA=T+timing=prepacked-compute'; do
  report "$candidate" "case=$value"
  reject logical
done

# Vendors use column-major storage even when the requested case has a packed recipe.
report "$candidate" implementation=openblas actual_kernel=vendor-cblas
accept logical
reject fixed
report "$base" timing_mode=raw-tile
report "$candidate" implementation=openblas actual_kernel=vendor-cblas timing_mode=raw-tile
reject logical

# Invalid or ambiguous recipes cannot produce a fixed comparison.
report "$base"
for value in 'gemm-block+15x7x31+uniform+packed=16x4' \
  'gemm-block+15x7x31+uniform+packed=4x4+packed=8x4' 'gemm-block+shape+uniform'; do
  report "$candidate" "case=$value"
  reject fixed
done

# A timing selector excludes incompatible raw calls; repeated selectors retain both block modes.
report "$base"
row timing_mode=raw-tile >>"$base"
report "$candidate"
row timing_mode=vendor-arithmetic >>"$candidate"
reject logical
accept logical --timing prepacked-compute
lines 2
row timing_mode=pack-plus-compute >>"$base"
row timing_mode=pack-plus-compute >>"$candidate"
accept logical --timing prepacked-compute --timing pack-plus-compute
lines 3

# Policy rows cannot become fixed experiments merely by agreeing with themselves.
report "$base"
row case=gemm+15x7x31+uniform >>"$base"
cp "$base" "$candidate"
accept fixed
lines 2
report "$base" case=gemm+15x7x31+uniform
cp "$base" "$candidate"
reject fixed

# Disjoint, unsupported-only and historical reports cannot silently pass strict comparison.
report "$base"
for patch in case=gemm-block+16x7x31+uniform+packed=4x4+timing=prepacked-compute status=unsupported; do
  report "$candidate" "$patch"
  reject logical
done

report "$candidate"
sed -i '1s/run,id/run,unknown/' "$candidate"
reject logical

# Quoted CSV fields round-trip, while sample aggregation uses medians and extrema.
report "$base" 'actual_kernel=quoted "kernel", one' ns_per_op=1
row 'actual_kernel=quoted "kernel", one' ns_per_op=3 >>"$base"
report "$candidate" 'actual_kernel=quoted "kernel", one' ns_per_op=2
row 'actual_kernel=quoted "kernel", one' ns_per_op=4 >>"$candidate"
accept fixed
grep -Fq '"quoted ""kernel"", one"' "$temporary/output.csv" || fail "quoted kernel did not round-trip"
grep -Eq ',2,3,2,4,0[.]66666[0-9]+$' "$temporary/output.csv" || fail "incorrect sample statistics"

# Embedded newlines remain one CSV field, and numeric-looking metadata matches textually.
report "$base" $'actual_kernel=first line\nsecond line'
cp "$base" "$candidate"
accept fixed
grep -Fq 'second line"' "$temporary/output.csv" || fail "multiline kernel did not round-trip"
report "$base" threads=1
report "$candidate" threads=01
reject fixed

# Invalid samples and malformed records fail before comparison.
report "$base"
for value in 0 -1 NaN Infinity 1e999 bad ''; do
  report "$candidate" "ns_per_op=$value"
  reject logical
done
report "$candidate"
echo 'extra,columns' >>"$candidate"
reject logical
printf '\n' >"$candidate"
reject logical

# Every candidate is checked separately, including a filename containing an equals sign.
report "$candidate"
report "$temporary/second=report.csv" case=gemm-block+15x7x31+uniform+packed=8x4+timing=prepacked-compute
"$tools/compare.sh" --mode logical --require-compatible "$base" "$candidate" "$temporary/second=report.csv" >"$temporary/output.csv"
lines 3
if "$tools/compare.sh" "$base" "$candidate" >/dev/null 2>&1; then fail "missing comparison mode accepted"; fi

# Broken references, duplicate samples and missing measurements are rejected.
report "$candidate"
sed -i 's/^sample,1,/sample,99,/' "$candidate"
reject logical
report "$candidate"
tail -1 "$candidate" >>"$candidate"
reject logical
report "$candidate"
sed -i '$d' "$candidate"
reject logical
report "$candidate"
sed -i 's/^case,1,1,/case,1,99,/' "$candidate"
reject logical

echo 'Comparator shell tests passed'
