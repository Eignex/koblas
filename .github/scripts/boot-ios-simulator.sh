#!/usr/bin/env bash
# Boot independently of Gradle so toolchain setup and linking can use the startup wait.
set -euo pipefail

directory=${1:?simulator state directory is required}
trap 'printf "%s\n" "$?" > "$directory/status"' EXIT
test "$(uname -m)" = arm64
devices=$(xcrun simctl list devices available -j)
selection=$(jq -er '
  [.devices | to_entries[] | select(.key | contains(".iOS-")) | .value[] |
    select(.name | startswith("iPhone"))] |
  (map(select(.state == "Booted"))[0] // .[0] // error("No installed iPhone simulator")) |
  [.udid, .state] | @tsv' <<<"$devices")
read -r device state <<<"$selection"
printf '%s\n' "$device" > "$directory/device"
if [[ "$state" != Booted ]]; then xcrun simctl boot "$device"; fi
xcrun simctl bootstatus "$device" -b
