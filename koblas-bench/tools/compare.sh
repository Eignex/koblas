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
  required_count = split("schema workload_version fixture_version timing_mode threads warmups target_ns comparison_kind configuration physical_work case logical_id actual_kernel implementation status ns_per_op", required)
  match_count = split("schema workload_version fixture_version timing_mode threads warmups target_ns comparison_kind", match_fields)
  group_count_fields = split("logical_id schema workload_version fixture_version timing_mode threads warmups target_ns comparison_kind configuration physical_work implementation actual_kernel", group_fields)
  timing_count = split(timings, timing_values, "\034")
  for (i = 1; i <= timing_count; i++) if (timing_values[i] != "") selected_timing[timing_values[i]] = 1
}

function fail(message) {
  print message > "/dev/stderr"
  failed = 1
  exit 1
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

function compatible(left, right,    i, field) {
  if (("" rows[left]["logical_id"]) != ("" rows[right]["logical_id"])) return 0
  for (i = 1; i <= match_count; i++) {
    field = match_fields[i]
    if (("" rows[left][field]) != ("" rows[right][field])) return 0
  }
  if (mode == "fixed" || rows[left]["timing_mode"] ~ /^(raw-tile|packing-only|layout-only|vendor-arithmetic)$/) {
    return ("" rows[left]["configuration"]) == ("" rows[right]["configuration"]) && ("" rows[left]["physical_work"]) == ("" rows[right]["physical_work"])
  }
  return 1
}

BEGINFILE {
  file_count = ARGIND
  names[ARGIND] = FILENAME
  delete header
}

FNR == 1 {
  sub(/\r$/, "")
  columns = read_csv($0, fields)
  if (columns < 0) fail(FILENAME ": malformed benchmark CSV header")
  for (i = 1; i <= columns; i++) {
    if (fields[i] in header) fail(FILENAME ": duplicate benchmark CSV column")
    header[fields[i]] = i
  }
  for (i = 1; i <= required_count; i++) {
    if (!(required[i] in header)) fail(FILENAME ": incompatible or empty benchmark CSV")
  }
  next
}

{
  sub(/\r$/, "")
  if ($0 == "") next
  if (read_csv($0, fields) != columns) fail(FILENAME ": malformed benchmark CSV")
  delete row
  for (i = 1; i <= required_count; i++) {
    field = required[i]
    row[field] = fields[header[field]]
    if (field != "ns_per_op" && row[field] == "") fail(FILENAME ": malformed benchmark CSV")
  }
  if (row["schema"] != "4") fail(FILENAME ": unsupported schema; capture a fresh baseline")
  if (row["status"] != "ok") next
  value = row["ns_per_op"]
  if (value !~ /^[+]?[0-9]*([.][0-9]+|[0-9]+[.]?[0-9]*)([eE][+-]?[0-9]+)?$/ || value + 0 <= 0 || tolower(sprintf("%g", value + 0)) ~ /inf|nan/) fail(FILENAME ": invalid sample")
  successful[ARGIND]++
  if (length(selected_timing) && !(row["timing_mode"] in selected_timing)) next
  if (mode == "fixed" && row["configuration"] == "policy-v1") next

  # Length prefixes keep physical strategies distinct without delimiter collisions in CSV values.
  key = ""
  for (i = 1; i <= group_count_fields; i++) {
    value = row[group_fields[i]]
    key = key length(value) ":" value
  }
  if (!(key in groups[ARGIND])) {
    group = ++group_count
    groups[ARGIND][key] = group
    order[ARGIND][++group_counts[ARGIND]] = group
    for (field in row) rows[group][field] = row[field]
    identities[ARGIND][row["logical_id"]] = 1
  }
  group = groups[ARGIND][key]
  samples[group][++sample_counts[group]] = row["ns_per_op"] + 0
}

ENDFILE {
  if (!successful[ARGIND]) fail(FILENAME ": no successful measurements")
}

END {
  if (failed) exit 1
  for (group = 1; group <= group_count; group++) {
    medians[group] = median(samples[group], sample_counts[group])
    asort(samples[group], samples[group], "@val_num_asc")
  }
  print "candidate,logical_id,base_case,candidate_case,base_configuration,candidate_configuration,base_physical_work,candidate_physical_work,base_kernel,candidate_kernel,base_median_ns,candidate_median_ns,candidate_min_ns,candidate_max_ns,base_over_candidate"
  for (candidate = 2; candidate <= file_count; candidate++) {
    delete joined
    pairs = 0
    for (left_index = 1; left_index <= group_counts[1]; left_index++) {
      left = order[1][left_index]
      for (right_index = 1; right_index <= group_counts[candidate]; right_index++) {
        right = order[candidate][right_index]
        if (!compatible(left, right)) continue
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%.17g,%.17g,%.17g,%.17g,%.17g\n", csv(names[candidate]), csv(rows[left]["logical_id"]), csv(rows[left]["case"]), csv(rows[right]["case"]), csv(rows[left]["configuration"]), csv(rows[right]["configuration"]), csv(rows[left]["physical_work"]), csv(rows[right]["physical_work"]), csv(rows[left]["actual_kernel"]), csv(rows[right]["actual_kernel"]), medians[left], medians[right], samples[right][1], samples[right][sample_counts[right]], medians[left] / medians[right]
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
