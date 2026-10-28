#!/usr/bin/env bash
set -euo pipefail

next_patch() {
  local tag major minor patch
  while IFS= read -r tag; do
    if [[ "$tag" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
      major=${BASH_REMATCH[1]}
      minor=${BASH_REMATCH[2]}
      patch=${BASH_REMATCH[3]}
      printf '%s.%s.%s\n' "$major" "$minor" "$((10#$patch + 1))"
      return
    fi
  done < <(git tag --list 'v*' --sort=-v:refname)
  echo '0.0.1'
}

dry_run=${DRY_RUN:-false}
if [[ "$dry_run" == true ]]; then
  version=$(next_patch)
elif [[ "${GITHUB_REF_TYPE:-branch}" == tag ]]; then
  version=${GITHUB_REF_NAME#v}
else
  version="$(next_patch)-SNAPSHOT"
fi

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "::error::invalid publication version: $version" >&2
  exit 1
fi

is_snapshot=false
[[ "$version" != *-SNAPSHOT ]] || is_snapshot=true
is_prerelease=false
[[ "$version" != *-* ]] || is_prerelease=true
{
  echo "version=$version"
  echo "is_snapshot=$is_snapshot"
  echo "is_prerelease=$is_prerelease"
  echo "dry_run=$dry_run"
  echo "revision=$(git rev-parse HEAD)"
} >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
