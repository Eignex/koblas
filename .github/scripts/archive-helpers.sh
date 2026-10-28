#!/usr/bin/env bash

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | awk '{print $1}'
}

validate_archive() {
  local archive=$1 entries=$2 name
  unzip -tq "$archive" >/dev/null || return 1
  unzip -Z -1 "$archive" >"$entries" || return 1
  if [[ -n $(LC_ALL=C sort "$entries" | uniq -d) ]]; then
    echo 'duplicate archive entries' >&2; return 1
  fi
  while IFS= read -r name; do
    case "/$name/" in
      //*|*/../*|*\\*) echo "unsafe artifact path: $name" >&2; return 1 ;;
    esac
  done <"$entries"
  # A symlink could redirect a later entry outside the extraction directory.
  unzip -Z -l "$archive" >"$entries.details" || return 1
  if grep -q '^l' "$entries.details"; then
    echo 'unsafe artifact symlink' >&2; return 1
  fi
}

extract_verified_archive() (
  set -euo pipefail
  local archive=$1 digest=$2 destination=$3 temporary
  [[ $digest == sha256:* ]] || { echo 'artifact has no SHA-256 digest' >&2; exit 1; }
  [[ $digest == "sha256:$(sha256 "$archive")" ]] || { echo 'artifact digest mismatch' >&2; exit 1; }
  temporary=$(mktemp -d)
  trap 'rm -rf "$temporary"' EXIT
  validate_archive "$archive" "$temporary/entries" || exit 1
  # Extract into a fresh directory so existing destination symlinks cannot redirect unzip.
  mkdir "$temporary/extracted"
  unzip -q "$archive" -d "$temporary/extracted"
  if [[ -e $destination ]]; then
    [[ -d $destination && ! -L $destination && -z $(find "$destination" -mindepth 1 -print -quit) ]] || {
      echo "artifact destination must be empty: $destination" >&2; exit 1;
    }
  else
    mkdir -p "$destination"
  fi
  cp -R "$temporary/extracted/." "$destination/"
)
