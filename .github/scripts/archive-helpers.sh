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
