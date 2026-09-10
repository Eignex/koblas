#!/usr/bin/env bash
set -euo pipefail

require_compatible=false
if [[ ${1-} == --require-compatible ]]; then
  require_compatible=true
  shift
fi

if (( $# < 2 )); then
  echo "usage: compare.sh [--require-compatible] BASE.csv CANDIDATE.csv [CANDIDATE.csv ...]" >&2
  exit 2
fi

command -v gawk >/dev/null || {
  echo "compare.sh requires GNU awk (gawk)" >&2
  exit 1
}

gawk -v require_compatible="$require_compatible" '
BEGIN {
  separator = sprintf("%c", 29)
  required_count = split("case implementation workload_version fixture_version ns_per_op status comparison_kind timing_mode threads warmups target_ns", required)
  match_count = split("workload_version fixture_version timing_mode threads comparison_kind warmups target_ns", match_fields)
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
  if (state == 1) return -1
  fields[output_index] = field
  return output_index
}

function csv(value) {
  if (value ~ /[",\r\n]/) {
    gsub(/"/, "\"\"", value)
    return "\"" value "\""
  }
  return value
}

function median(values, count,    ordered) {
  delete ordered
  for (value_index = 1; value_index <= count; value_index++) ordered[value_index] = values[value_index]
  asort(ordered, ordered, "@val_num_asc")
  if (count % 2) return ordered[(count + 1) / 2]
  return (ordered[count / 2] + ordered[count / 2 + 1]) / 2
}

function python_quote(value) {
  gsub(/\\/, "\\\\", value)
  gsub(/\047/, "\\\047", value)
  return "\047" value "\047"
}

function metadata_list(file_index, case_name,    count, item, list, ordered) {
  delete ordered
  count = 0
  for (item in metadata) {
    split(item, parts, SUBSEP)
    if (parts[1] == file_index && parts[2] == case_name) ordered[++count] = parts[3]
  }
  asort(ordered)
  list = "["
  for (item = 1; item <= count; item++) list = list (item == 1 ? "" : ", ") python_quote(ordered[item])
  return list "]"
}

FNR == 1 {
  sub(/\r$/, "")
  file_index = ++file_count
  file_number[FILENAME] = file_index
  field_count = parse_csv($0, fields)
  if (field_count < 0) fail(FILENAME ": malformed benchmark CSV header")
  for (field_index = 1; field_index <= field_count; field_index++) header[file_index, fields[field_index]] = field_index
  for (required_index = 1; required_index <= required_count; required_index++) {
    if (!((file_index SUBSEP required[required_index]) in header)) fail(FILENAME ": incompatible or empty benchmark CSV")
  }
  next
}

{
  sub(/\r$/, "")
  if ($0 == "") next
  file_index = file_number[FILENAME]
  field_count = parse_csv($0, fields)
  if (field_count < 0) fail(FILENAME ": malformed benchmark CSV")
  row_count[file_index]++
  case_name = fields[header[file_index, "case"]]
  metadata_value = ""
  full_key = case_name
  for (match_index = 1; match_index <= match_count; match_index++) {
    value = fields[header[file_index, match_fields[match_index]]]
    full_key = full_key separator value
    metadata_value = metadata_value (match_index == 1 ? "" : "/") value
  }
  keys[file_index, full_key] = 1
  metadata[file_index, case_name, metadata_value] = 1
  if (fields[header[file_index, "status"]] == "ok") {
    successful_cases[file_index, case_name] = 1
    successful_count[file_index, full_key]++
    successful_ns[file_index, full_key, successful_count[file_index, full_key]] = fields[header[file_index, "ns_per_op"]] + 0
    comparison_kind[file_index, full_key] = fields[header[file_index, "comparison_kind"]]
  }
}

END {
  if (failed) exit 1
  for (file_index = 1; file_index <= file_count; file_index++) {
    if (!row_count[file_index]) fail(ARGV[file_index] ": incompatible or empty benchmark CSV")
  }

  incompatible = 0
  for (candidate_index = 2; candidate_index <= file_count; candidate_index++) {
    print "candidate=" ARGV[candidate_index]
    print "case,base_median_ns,candidate_median_ns,candidate_min_ns,candidate_max_ns,base_over_candidate,comparison_kind"
    delete common_keys
    for (compound in keys) {
      split(compound, parts, SUBSEP)
      if (parts[1] == 1 && ((candidate_index SUBSEP parts[2]) in keys)) common_keys[parts[2]] = 1
    }
    common_count = asorti(common_keys, ordered_keys)
    for (key_index = 1; key_index <= common_count; key_index++) {
      full_key = ordered_keys[key_index]
      left_count = successful_count[1, full_key]
      right_count = successful_count[candidate_index, full_key]
      if (!left_count || !right_count) continue
      delete left
      delete right
      for (sample = 1; sample <= left_count; sample++) left[sample] = successful_ns[1, full_key, sample]
      for (sample = 1; sample <= right_count; sample++) right[sample] = successful_ns[candidate_index, full_key, sample]
      left_median = median(left, left_count)
      right_median = median(right, right_count)
      asort(right, right, "@val_num_asc")
      split(full_key, key_parts, separator)
      printf "%s,%.9g,%.9g,%.9g,%.9g,%.6g,%s\n", csv(key_parts[1]), left_median, right_median, right[1], right[right_count], left_median / right_median, csv(comparison_kind[candidate_index, full_key])
      joined[candidate_index, key_parts[1]] = 1
    }

    delete incompatible_cases
    for (compound in successful_cases) {
      split(compound, parts, SUBSEP)
      if (parts[1] == 1 && ((candidate_index SUBSEP parts[2]) in successful_cases) && !((candidate_index SUBSEP parts[2]) in joined)) incompatible_cases[parts[2]] = 1
    }
    incompatible_count = asorti(incompatible_cases, ordered_cases)
    for (case_index = 1; case_index <= incompatible_count; case_index++) {
      case_name = ordered_cases[case_index]
      print "incompatible case=" case_name " base=" metadata_list(1, case_name) " candidate=" metadata_list(candidate_index, case_name) > "/dev/stderr"
      incompatible++
    }
  }
  if (require_compatible == "true" && incompatible) {
    print incompatible " supported case pairs have incompatible metadata" > "/dev/stderr"
    exit 1
  }
}
' "$@"
