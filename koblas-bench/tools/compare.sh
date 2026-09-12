#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: compare.sh --mode fixed|logical [--require-compatible] [--timing BOUNDARY] BASE.csv CANDIDATE.csv [CANDIDATE.csv ...]" >&2
  exit 2
}

require_compatible=false
mode=
timings=
files=()
while (( $# )); do
  case $1 in
    --require-compatible) require_compatible=true; shift ;;
    --mode)
      (( $# >= 2 )) || usage
      mode=$2
      shift 2
      ;;
    --timing)
      (( $# >= 2 )) && [[ -n $2 ]] || usage
      timings+="${2}"$'\034'
      shift 2
      ;;
    --) shift; files+=("$@"); break ;;
    -*) usage ;;
    *) files+=("$1"); shift ;;
  esac
done
[[ $mode == fixed || $mode == logical ]] || usage
(( ${#files[@]} >= 2 )) || usage
command -v gawk >/dev/null || {
  echo "compare.sh requires GNU awk (gawk)" >&2
  exit 1
}
# Prefix relative paths so awk cannot interpret a filename containing = as an assignment.
for i in "${!files[@]}"; do
  [[ ${files[i]} == /* ]] || files[i]="./${files[i]}"
done

gawk -v require_compatible="$require_compatible" -v mode="$mode" -v timings="$timings" '
BEGIN {
  record_type[1] = "run"; record_type[2] = "case"; record_type[3] = "sample"
  expected["run"] = "id implementation pass unit source_commit dirty runtime threads warmups target_ns harness warmup_target_ns forks"
  expected["case"] = "id run_id case status comparison_kind timing_mode actual_kernel"
  expected["sample"] = "case_id fork sample operations elapsed_ns ns_per_op"
  for (type in expected) column_count[type] = split(expected[type], columns[type]) + 1
  match_count = split("timing_mode threads warmups target_ns comparison_kind", match_fields)
  group_count_fields = split("logical_id timing_mode threads warmups target_ns comparison_kind operation configuration implementation actual_kernel", group_fields)
  timing_count = split(timings, timing_values, "\034")
  for (i = 1; i <= timing_count; i++) if (timing_values[i] != "") selected_timing[timing_values[i]] = 1
}

function fail(message) {
  print message > "/dev/stderr"
  failed = 1
  exit 1
}

function positive(value) {
  return value ~ /^[+]?[0-9]*([.][0-9]+|[0-9]+[.]?[0-9]*)([eE][+-]?[0-9]+)?$/ && value + 0 > 0 && tolower(sprintf("%g", value + 0)) !~ /inf|nan/
}

function parse_csv(line, fields,    character, field, length_, next_character, output_index, position, state) {
  delete fields
  field = ""
  output_index = 1
  state = 0
  length_ = length(line)
  for (position = 1; position <= length_; position++) {
    character = substr(line, position, 1)
    if (state == 1) {
      if (character == "\"") {
        next_character = substr(line, position + 1, 1)
        if (next_character == "\"") {
          field = field "\""
          position++
        } else {
          state = 2
        }
      } else {
        field = field character
      }
    } else if (state == 2) {
      if (character != ",") return -1
      fields[output_index++] = field
      field = ""
      state = 0
    } else if (character == ",") {
      fields[output_index++] = field
      field = ""
    } else if (character == "\"" && field == "") {
      state = 1
    } else if (character == "\"") {
      return -1
    } else {
      field = field character
    }
  }
  if (state == 1) return -2
  fields[output_index] = field
  return output_index
}

function read_csv(line, fields,    count, continuation) {
  count = parse_csv(line, fields)
  while (count == -2) {
    if ((getline continuation) <= 0) return -1
    sub(/\r$/, "", continuation)
    line = line "\n" continuation
    count = parse_csv(line, fields)
  }
  return count
}

function csv(value) {
  if (value ~ /[",\r\n]/) {
    gsub(/"/, "\"\"", value)
    return "\"" value "\""
  }
  return value
}

function median(values, count,    ordered, value_index) {
  delete ordered
  for (value_index = 1; value_index <= count; value_index++) ordered[value_index] = values[value_index]
  asort(ordered, ordered, "@val_num_asc")
  if (count % 2) return ordered[(count + 1) / 2]
  return ordered[count / 2] / 2 + ordered[count / 2 + 1] / 2
}

# Shape and recipe determine physical work; the source commit identifies the implementation.
function identify_case(row,    parts, count, operation, logical, options, i, key, value, names, n) {
  count = split(row["case"], parts, "+")
  if (count < 3 || parts[2] !~ /^[1-9][0-9]*(x[1-9][0-9]*)*$/) fail(FILENAME ": malformed case")
  operation = parts[1]
  row["operation"] = operation
  logical = operation == "gemm-tile" || operation == "gemm-block" ? "gemm-add" : \
      operation == "packed-trsm" ? "right-solve" : operation == "gemm-trsm" ? "update-solve" : operation
  logical = logical "+" parts[2] "+" parts[3]
  row["configuration"] = "policy"
  for (i = 4; i <= count; i++) {
    n = index(parts[i], "=")
    if (n < 2 || n == length(parts[i])) fail(FILENAME ": malformed case option")
    key = substr(parts[i], 1, n - 1); value = substr(parts[i], n + 1)
    if (key in options) fail(FILENAME ": duplicate case option")
    options[key] = value
    if (key == "packed") {
      if (value != "4x4" && value != "8x4") fail(FILENAME ": unsupported packed recipe")
      row["configuration"] = row["implementation"] ~ /^(openblas|onemkl|accelerate)$/ ? "column-major" : "packed=" value
    }
  }
  n = asorti(options, names)
  for (i = 1; i <= n; i++) {
    key = names[i]
    if (key != "packed" && key != "timing") logical = logical "+" key "=" options[key]
  }
  row["logical_id"] = logical
}

function compatible(left, right,    i, field) {
  if (("" rows[left]["logical_id"]) != ("" rows[right]["logical_id"])) return 0
  for (i = 1; i <= match_count; i++) {
    field = match_fields[i]
    if (("" rows[left][field]) != ("" rows[right][field])) return 0
  }
  if (mode == "fixed" || rows[left]["timing_mode"] ~ /^(raw-tile|packing-only|layout-only|vendor-arithmetic)$/) {
    return ("" rows[left]["configuration"]) == ("" rows[right]["configuration"]) && ("" rows[left]["operation"]) == ("" rows[right]["operation"])
  }
  return 1
}

BEGINFILE {
  compact = 0
  column_count["case"] = split(expected["case"], columns["case"]) + 1
  file_count = ARGIND
  names[ARGIND] = FILENAME
  delete runs
  delete cases
  delete case_samples
  delete seen_samples
}

FNR <= 2 || (FNR == 3 && !compact) {
  sub(/\r$/, "")
  type = record_type[FNR]
  count = read_csv($0, fields)
  if (FNR == 2 && count == 13) {
    compact = 1
    column_count["case"] = split(expected["case"] " samples forks median_ns min_ns max_ns", columns["case"]) + 1
  }
  if (count != column_count[type] || fields[1] != type) fail(FILENAME ": malformed benchmark CSV header")
  for (i = 2; i <= column_count[type]; i++) {
    if (fields[i] != columns[type][i - 1]) fail(FILENAME ": malformed benchmark CSV header")
  }
  next
}

{
  sub(/\r$/, "")
  if ($0 == "") next
  count = read_csv($0, fields)
  type = fields[1]
  if (!(type in column_count) || count != column_count[type]) fail(FILENAME ": malformed benchmark CSV record")
  delete record
  for (i = 2; i <= count; i++) {
    if (fields[i] == "" && !(compact && type == "case" && i >= 11)) fail(FILENAME ": empty benchmark CSV field")
    record[columns[type][i - 1]] = fields[i]
  }
  if (type == "run") {
    id = record["id"]
    if (id !~ /^[1-9][0-9]*$/ || (id in runs)) fail(FILENAME ": invalid or duplicate run id")
    for (field in record) if (field != "id") runs[id][field] = record[field]
    next
  }
  if (type == "case") {
    id = record["id"]
    if (id !~ /^[1-9][0-9]*$/ || (id in cases) || !(record["run_id"] in runs)) fail(FILENAME ": invalid case or unknown run id")
    for (field in record) if (field != "id") cases[id][field] = record[field]
    if (!compact) next
    if (record["status"] != "ok") {
      if (record["status"] != "unsupported" || record["samples"] != "0" || record["forks"] != "0" || record["median_ns"] != "" || record["min_ns"] != "" || record["max_ns"] != "") fail(FILENAME ": invalid unmeasured summary")
      next
    }
    if (record["samples"] !~ /^[1-9][0-9]*$/ || record["forks"] !~ /^[1-9][0-9]*$/ || record["forks"] + 0 != runs[record["run_id"]]["forks"] + 0 || record["samples"] + 0 < record["forks"] + 0) fail(FILENAME ": invalid summary counts")
    if (!positive(record["median_ns"]) || !positive(record["min_ns"]) || !positive(record["max_ns"]) || record["min_ns"] + 0 > record["median_ns"] + 0 || record["median_ns"] + 0 > record["max_ns"] + 0) fail(FILENAME ": invalid summary statistics")
    record["ns_per_op"] = record["median_ns"]
    case_samples[id] = record["samples"]
  } else {
    if (compact) fail(FILENAME ": sample record in compact report")
    id = record["case_id"]
    if (!(id in cases) || cases[id]["status"] != "ok") fail(FILENAME ": sample references unknown or unsupported case")
    for (i = 1; i <= 4; i++) {
      field = columns["sample"][i + 1]
      if (record[field] !~ /^[1-9][0-9]*$/) fail(FILENAME ": invalid sample " field)
    }
    key = id SUBSEP record["fork"] SUBSEP record["sample"]
    if (key in seen_samples) fail(FILENAME ": duplicate sample")
    seen_samples[key] = 1
    case_samples[id]++
  }
  delete row
  for (field in runs[cases[id]["run_id"]]) row[field] = runs[cases[id]["run_id"]][field]
  for (field in cases[id]) row[field] = cases[id][field]
  for (field in record) row[field] = record[field]
  identify_case(row)
  value = row["ns_per_op"]
  if (!positive(value)) fail(FILENAME ": invalid sample")
  successful[ARGIND]++
  if (length(selected_timing) && !(row["timing_mode"] in selected_timing)) next
  if (mode == "fixed" && row["configuration"] == "policy") next

  # Length prefixes keep physical strategies distinct without delimiter collisions in CSV values.
  key = ""
  for (i = 1; i <= group_count_fields; i++) {
    value = row[group_fields[i]]
    key = key length(value) ":" value
  }
  # Summary medians cannot reconstruct pooled distributions; keep each case record separate.
  if (compact) key = key ":case=" id
  if (!(key in groups[ARGIND])) {
    group = ++group_count
    groups[ARGIND][key] = group
    order[ARGIND][++group_counts[ARGIND]] = group
    for (field in row) rows[group][field] = row[field]
    identities[ARGIND][row["logical_id"]] = 1
  }
  group = groups[ARGIND][key]
  if (compact) {
    summarized[group] = 1
    medians[group] = row["median_ns"] + 0
    minimums[group] = row["min_ns"] + 0
    maximums[group] = row["max_ns"] + 0
  } else samples[group][++sample_counts[group]] = row["ns_per_op"] + 0
}

ENDFILE {
  for (id in cases) if (cases[id]["status"] == "ok" && !case_samples[id]) fail(FILENAME ": supported case has no samples")
  if (!successful[ARGIND]) fail(FILENAME ": no successful measurements")
}

END {
  if (failed) exit 1
  for (group = 1; group <= group_count; group++) {
    if (summarized[group]) continue
    medians[group] = median(samples[group], sample_counts[group])
    asort(samples[group], samples[group], "@val_num_asc")
    minimums[group] = samples[group][1]
    maximums[group] = samples[group][sample_counts[group]]
  }
  print "candidate,logical_id,base_case,candidate_case,base_configuration,candidate_configuration,base_kernel,candidate_kernel,base_median_ns,candidate_median_ns,candidate_min_ns,candidate_max_ns,base_over_candidate"
  for (candidate = 2; candidate <= file_count; candidate++) {
    delete joined
    pairs = 0
    for (left_index = 1; left_index <= group_counts[1]; left_index++) {
      left = order[1][left_index]
      for (right_index = 1; right_index <= group_counts[candidate]; right_index++) {
        right = order[candidate][right_index]
        if (!compatible(left, right)) continue
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", csv(names[candidate]), csv(rows[left]["logical_id"]), csv(rows[left]["case"]), csv(rows[right]["case"]), csv(rows[left]["configuration"]), csv(rows[right]["configuration"]), csv(rows[left]["actual_kernel"]), csv(rows[right]["actual_kernel"]), medians[left], medians[right], minimums[right], maximums[right], medians[left] / medians[right]
        joined[left] = joined[right] = 1
        pairs++
      }
    }
    if (!pairs) {
      print "incompatible case=no compatible measurements in " names[candidate] > "/dev/stderr"
      incompatible = 1
    }
    for (side = 1; side <= 2; side++) {
      file = side == 1 ? 1 : candidate
      peer = side == 1 ? candidate : 1
      for (i = 1; i <= group_counts[file]; i++) {
        group = order[file][i]
        if ((rows[group]["logical_id"] in identities[peer]) && !(group in joined)) {
          print "incompatible case=" rows[group]["case"] " side=" (side == 1 ? "base" : "candidate") " mode=" mode > "/dev/stderr"
          incompatible = 1
        }
      }
    }
  }
  if (require_compatible == "true" && incompatible) exit 1
}
' "${files[@]}"
