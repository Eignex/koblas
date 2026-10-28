#!/usr/bin/env bash
# Verify a run artifact's digest before extracting it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/archive-helpers.sh"

repository=${1:?usage: download-artifact.sh REPOSITORY RUN_ID NAME DESTINATION}
run_id=${2:?}
name=${3:?}
destination=${4:?}
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
gh api --paginate --slurp "repos/$repository/actions/runs/$run_id/artifacts" >"$temporary/pages.json"
jq --arg name "$name" '[.[].artifacts[] | select(.name == $name and (.expired | not))]' \
  "$temporary/pages.json" >"$temporary/artifacts.json"
count=$(jq length "$temporary/artifacts.json")
[[ $count == 1 ]] || { echo "::error::expected one unexpired artifact named $name, found $count" >&2; exit 1; }
id=$(jq -er '.[0].id' "$temporary/artifacts.json")
digest=$(jq -r '.[0].digest // empty' "$temporary/artifacts.json")
gh api "repos/$repository/actions/artifacts/$id/zip" >"$temporary/artifact.zip"
extract_verified_archive "$temporary/artifact.zip" "$digest" "$destination"
echo "Downloaded and verified $name: $digest"
