#!/usr/bin/env bash
set -euo pipefail
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
reject() {
  if "$@" >"$temporary/rejected.log" 2>&1; then echo "Expected failure: $*" >&2; exit 1; fi
}
mkdir "$temporary/reports"
reject bash "$scripts/verify-test-results.sh" "$temporary/reports"
for attributes in 'tests="0"' 'tests="2" failures="1"' 'tests="2" errors="1"' 'tests="2" skipped="2"' 'tests="invalid"'; do
  echo "<testsuite $attributes/>" >"$temporary/reports/TEST-fixture.xml"
  reject bash "$scripts/verify-test-results.sh" "$temporary/reports"
done
echo '<testsuite tests="2" failures="0" errors="0" skipped="1"/>' >"$temporary/reports/TEST-fixture.xml"
bash "$scripts/verify-test-results.sh" "$temporary/reports"

jar="$temporary/jar"
mkdir -p "$jar/com/eignex/koblas" "$jar/META-INF/versions/25/com/eignex/koblas"
printf 'Manifest-Version: 1.0\nMulti-Release: true\n' >"$jar/META-INF/MANIFEST.MF"
printf '\312\376\272\276\000\000\000\075' >"$jar/com/eignex/koblas/JvmBackend.class"
printf '\312\376\272\276\000\000\000\105' >"$jar/META-INF/versions/25/com/eignex/koblas/JvmBackend.class"
metadata="$temporary/module.json"
echo '{"variants":[{"attributes":{"org.gradle.category":"library","org.gradle.jvm.version":17}}]}' >"$metadata"
verify() {
  rm -f "$temporary/library.jar"
  (cd "$jar" && zip -qr "$temporary/library.jar" .)
  bash "$scripts/verify-jvm-artifact.sh" "$temporary/library.jar" "$metadata" "$@"
}
verify
echo 'java/lang/foreign/' >>"$jar/com/eignex/koblas/JvmBackend.class"
reject verify
printf '\312\376\272\276\000\000\000\075' >"$jar/com/eignex/koblas/JvmBackend.class"
echo '{}' >"$jar/native.so"
reject verify
rm "$jar/native.so"
echo '{"variants":[{"attributes":{"org.gradle.category":"library","org.gradle.jvm.version":25}}]}' >"$metadata"
reject verify
echo '{"variants":[]}' >"$metadata"
apk="$temporary/apk"
mkdir "$apk"
echo portable >"$apk/classes.dex"
package_apk() { rm -f "$temporary/consumer.apk"; (cd "$apk" && zip -qr "$temporary/consumer.apk" .); }
package_apk
verify "$temporary/consumer.apk"
echo 'jdk/incubator/vector/' >>"$apk/classes.dex"
package_apk
reject verify "$temporary/consumer.apk"
echo portable >"$apk/classes.dex"
mkdir "$apk/lib"
echo native >"$apk/lib/native.so"
package_apk
reject verify "$temporary/consumer.apk"
echo 'Consumer verification checks passed'
