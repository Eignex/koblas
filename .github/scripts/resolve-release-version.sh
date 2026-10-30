#!/usr/bin/env bash
set -euo pipefail

core='(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'
is_snapshot=false
test_release=false
if [[ "$GITHUB_REF_TYPE" == tag ]]; then
  if [[ ! "$GITHUB_REF_NAME" =~ ^v${core}(-rc\.(0|[1-9][0-9]*))?$ ]]; then
    echo '::error::Use a vX.Y.Z or vX.Y.Z-rc.N release tag' >&2
    exit 1
  fi
  version=${GITHUB_REF_NAME#v}
  if [[ "$version" == *-rc.* ]]; then test_release=true; fi
elif [[ "$GITHUB_REF_TYPE" == branch && "$GITHUB_REF_NAME" == main ]]; then
  version=0.0.1-SNAPSHOT
  while IFS= read -r tag; do
    if [[ "$tag" =~ ^v${core}$ ]]; then
      version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))-SNAPSHOT"
      break
    fi
  done < <(git tag --list 'v*' --sort=-v:refname)
  is_snapshot=true
else
  echo '::error::Releases require a version tag; snapshots use main' >&2
  exit 1
fi
{
  echo "version=$version"
  echo "is_snapshot=$is_snapshot"
  echo "test_release=$test_release"
  echo "revision=$(git rev-parse HEAD)"
} >> "$GITHUB_OUTPUT"
