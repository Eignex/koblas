#!/usr/bin/env bash
# Check that prebuilt consumer runs and asynchronous simulator boot cannot report false success.
set -euo pipefail
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
fail() { echo "$*" >&2; exit 1; }
reject() {
  if "$@" >"$temporary/rejected.log" 2>&1; then fail "Expected rejection: $*"; fi
}
fixture() {
  cat <<EOF
INSTRUMENTATION_STATUS: class=com.eignex.koblas.consumer.JvmArtifactTest
INSTRUMENTATION_STATUS: test=portable & "<reference>"
INSTRUMENTATION_STATUS_CODE: 1
INSTRUMENTATION_STATUS: class=com.eignex.koblas.consumer.JvmArtifactTest
INSTRUMENTATION_STATUS: test=portable & "<reference>"
INSTRUMENTATION_STATUS: stack=assertion <failed> & stopped
INSTRUMENTATION_STATUS_CODE: $1
Time: 0.123
INSTRUMENTATION_CODE: $2
EOF
}
report() {
  awk -v suite=debug -f "$scripts/android-instrumentation-report.awk" "$1" >"$temporary/report.xml"
}
fixture 0 -1 >"$temporary/output"
report "$temporary/output"
xmllint --nonet --noout "$temporary/report.xml"
[[ $(xmllint --xpath 'string(/*/@tests)' "$temporary/report.xml") == 1 ]]
[[ $(xmllint --xpath 'string(/*/@time)' "$temporary/report.xml") == 0.123 ]]
[[ $(xmllint --xpath 'string(/*/testcase/@name)' "$temporary/report.xml") == 'portable & "<reference>"' ]]
for status in -1 -2 -3 -4; do
  fixture "$status" -1 >"$temporary/output"
  reject report "$temporary/output"
  case "$status" in -1) kind=error ;; -2) kind=failure ;; *) kind=skipped ;; esac
  [[ $(xmllint --xpath "count(/*/testcase/$kind)" "$temporary/report.xml") == 1 ]]
done
fixture 0 0 >"$temporary/output"
reject report "$temporary/output"
fixture 0 -1 | sed '/^INSTRUMENTATION_CODE:/d' >"$temporary/output"
reject report "$temporary/output"
fixture 1 -1 >"$temporary/output"
reject report "$temporary/output"
echo 'INSTRUMENTATION_CODE: -1' >"$temporary/output"
reject report "$temporary/output"
echo 'INSTRUMENTATION_FAILED: Process crashed.' >"$temporary/output"
reject report "$temporary/output"
fixture 0 -1 | sed '1,/^INSTRUMENTATION_STATUS_CODE: 1$/d' >"$temporary/output"
reject report "$temporary/output"
fixture -99 -1 >"$temporary/output"
reject report "$temporary/output"
fixture invalid -1 >"$temporary/output"
reject report "$temporary/output"

mkdir "$temporary/commands"
cat >"$temporary/commands/uname" <<'EOF'
#!/usr/bin/env bash
echo arm64
EOF
cat >"$temporary/commands/xcrun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "$*" >>"$SIMULATOR_FIXTURE/calls"
case "$*" in
  'simctl list devices available -j') cat "$SIMULATOR_FIXTURE/devices.json" ;;
  'simctl boot '*) if [[ $SIMULATOR_FAIL_BOOT == true ]]; then exit 7; fi ;;
esac
EOF
chmod +x "$temporary/commands/"*
boot_fixture() {
  local directory="$temporary/$1" devices=$2 failed=${3:-false}
  mkdir "$directory"
  jq -n --argjson devices "$devices" '{devices:{"com.apple.CoreSimulator.SimRuntime.iOS-26":$devices}}' \
    >"$directory/devices.json"
  PATH="$temporary/commands:$PATH" SIMULATOR_FIXTURE="$directory" SIMULATOR_FAIL_BOOT="$failed" \
    bash "$scripts/boot-ios-simulator.sh" "$directory"
}
boot_fixture ready '[{"name":"iPhone first","state":"Shutdown","udid":"first"},
                     {"name":"iPhone ready","state":"Booted","udid":"ready"}]'
[[ $(cat "$temporary/ready/device") == ready && $(cat "$temporary/ready/status") == 0 ]]
if grep -q '^simctl boot ' "$temporary/ready/calls"; then fail 'Booted simulator was restarted'; fi
grep -Fxq 'simctl bootstatus ready -b' "$temporary/ready/calls"
phone='[{"name":"iPhone test","state":"Shutdown","udid":"test"}]'
boot_fixture shutdown "$phone"
printf 'simctl list devices available -j\nsimctl boot test\nsimctl bootstatus test -b\n' >"$temporary/expected"
cmp "$temporary/expected" "$temporary/shutdown/calls"
reject boot_fixture failed "$phone" true
[[ $(cat "$temporary/failed/status") == 7 ]]
if grep -q bootstatus "$temporary/failed/calls"; then fail 'Waited on a failed simulator boot'; fi
reject boot_fixture missing '[]'
[[ $(cat "$temporary/missing/status") != 0 ]]

mkdir "$temporary/timing"
cat >"$temporary/timing/TEST-fixture.xml" <<'EOF'
<testsuite tests="2"><testcase name="quick" classname="Fixture" time="0.25"/>
<testcase name="slow" classname="Fixture" time="1.5"/></testsuite>
EOF
bash "$scripts/summarize-test-results.sh" "$temporary/timing" >"$temporary/timing.md"
grep -Fq '2 tests; 1.750 seconds' "$temporary/timing.md"
grep -Fq '| Fixture.slow | 1.500 |' "$temporary/timing.md"
echo 'Consumer CI checks passed'
