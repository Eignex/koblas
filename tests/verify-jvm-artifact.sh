#!/usr/bin/env bash
# Verify the JVM bytecode boundary and consumer compatibility metadata.
set -euo pipefail

jar=${1:?usage: verify-jvm-artifact.sh JAR MODULE_JSON [APK...]}
metadata=${2:?usage: verify-jvm-artifact.sh JAR MODULE_JSON [APK...]}
shift 2
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
fail() { echo "$*" >&2; exit 1; }

unzip -p "$jar" META-INF/MANIFEST.MF >"$temporary/manifest"
grep -q 'Multi-Release: true' "$temporary/manifest" || fail 'Missing Multi-Release manifest attribute'
unzip -Z -1 "$jar" >"$temporary/entries"
base_count=0
versioned_count=0
while IFS= read -r name; do
  case "$name" in *.so|*.a|*.dll|*.dylib) fail "Native library in JVM artifact: $name" ;; esac
  [[ $name == *.class ]] || continue
  unzip -p "$jar" "$name" >"$temporary/class"
  read -r high low < <(od -An -tu1 -j6 -N2 "$temporary/class")
  major=$((high * 256 + low))
  if [[ $name == META-INF/versions/25/* ]]; then
    [[ $major == 69 ]] || fail "Wrong desktop class version: $name ($major)"
    versioned_count=$((versioned_count + 1))
  else
    [[ $name != META-INF/versions/* || $name == META-INF/versions/17/module-info.class ]] || fail "Unexpected versioned class: $name"
    [[ $major == 61 ]] || fail "Wrong base class version: $name ($major)"
    if LC_ALL=C grep -aEq 'java/lang/foreign/|jdk/incubator/vector/|java/lang/management/|com/sun/management/' "$temporary/class"; then
      fail "Desktop API in base class: $name"
    fi
    base_count=$((base_count + 1))
  fi
done <"$temporary/entries"
grep -Fxq com/eignex/koblas/JvmBackend.class "$temporary/entries" || fail 'Missing base backend'
grep -Fxq META-INF/versions/25/com/eignex/koblas/JvmBackend.class "$temporary/entries" || fail 'Missing desktop backend'
((base_count && versioned_count)) || fail 'Missing base or desktop classes'
jq -e 'all(.variants[]; if .attributes["org.gradle.category"] == "library" then .attributes["org.gradle.jvm.version"] == 17 else true end)' "$metadata" >/dev/null
echo "Verified $base_count base and $versioned_count desktop classes; published JVM baseline is 17"

for apk in "$@"; do
  unzip -Z -1 "$apk" >"$temporary/entries"
  while IFS= read -r name; do
    [[ $name != lib/* ]] || fail "Native library in Android APK: $apk ($name)"
    if [[ $name == *.dex ]]; then
      unzip -p "$apk" "$name" >"$temporary/dex"
      if LC_ALL=C grep -aEq 'java/lang/foreign/|jdk/incubator/vector/' "$temporary/dex"; then
        fail "Desktop API in Android APK: $apk ($name)"
      fi
    fi
  done <"$temporary/entries"
  echo "Verified portable Android APK without native libraries: $apk"
done
