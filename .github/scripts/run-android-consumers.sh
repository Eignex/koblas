#!/usr/bin/env bash
# Test the same prebuilt APKs on each emulator without starting Gradle again.
set -euo pipefail
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
apks=${1:?usage: run-android-consumers.sh APK_DIRECTORY REPORT_DIRECTORY}
destination=${2:?}
package=com.eignex.koblas.consumer

run_variant() {
  local variant=$1 reports="$destination/$1"
  mkdir -p "$reports" || return 1
  adb uninstall "$package.test" >/dev/null 2>&1 || true
  adb uninstall "$package" >/dev/null 2>&1 || true
  adb install -t "$apks/$variant/app.apk" || return 1
  adb install -t "$apks/$variant/test.apk" || return 1
  if ! timeout 180 adb shell am instrument -w -r "$package.test/androidx.test.runner.AndroidJUnitRunner" \
      >"$reports/instrumentation.log" 2>&1; then
    cat "$reports/instrumentation.log" >&2
    return 1
  fi
  cat "$reports/instrumentation.log"
  awk -v suite="android-$variant" -f "$scripts/android-instrumentation-report.awk" \
    "$reports/instrumentation.log" >"$reports/TEST-consumer.xml"
}

result=0
for variant in debug release; do
  if ! run_variant "$variant"; then
    echo "::error::$variant consumer instrumentation failed" >&2
    result=1
  fi
done
exit "$result"
