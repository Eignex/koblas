#!/usr/bin/env bash
# Check coordinates, required artifacts, checksums and signatures before upload.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/archive-helpers.sh"

archive=${1:?usage: check-release-bundle.sh BUNDLE VERSION}
version=${2:?}
temporary=$(mktemp -d)
cleanup() {
  gpgconf --homedir "$temporary/gnupg" --kill all 2>/dev/null || true
  rm -rf "$temporary"
}
trap cleanup EXIT
fail() { echo "::error::$*" >&2; exit 1; }
artifacts=(koblas koblas-jvm koblas-linuxx64 koblas-linuxarm64 koblas-mingwx64 koblas-macosarm64 koblas-iosarm64 koblas-iossimulatorarm64)
validate_archive "$archive" "$temporary/entries" || fail 'invalid release archive'
grep -v '/$' "$temporary/entries" >"$temporary/files"
: >"$temporary/payloads"
while IFS= read -r name; do
  IFS=/ read -r group organization artifact found_version filename extra <<<"$name"
  [[ $group == com && $organization == eignex && -n $filename && -z $extra ]] || fail "unexpected publication: $name"
  extension=
  for expected in "${artifacts[@]}"; do
    if [[ $artifact == "$expected" ]]; then
      extension=klib
      [[ $artifact != koblas && $artifact != koblas-jvm ]] || extension=jar
    fi
  done
  [[ -n $extension ]] || fail "unexpected publication: $name"
  [[ $found_version == "$version" ]] || fail "wrong publication version: $name"
  payload=$filename
  case "$payload" in *.md5|*.sha1|*.sha256|*.sha512) payload=${payload%.*} ;; esac
  payload=${payload%.asc}
  base="$artifact-$version"
  case "$payload" in
    "$base.pom"|"$base.module"|"$base.$extension"|"$base-sources.jar"|"$base-javadoc.jar") ;;
    "$base-kotlin-tooling-metadata.json") [[ $artifact == koblas ]] || fail "unexpected artifact: $name" ;;
    "$base-metadata.jar") [[ $extension == klib ]] || fail "unexpected artifact: $name" ;;
    *) fail "unexpected artifact: $name" ;;
  esac
  [[ $payload != "$filename" ]] || echo "$name" >>"$temporary/payloads"
done <"$temporary/files"
mkdir "$temporary/bundle"
unzip -q "$archive" -d "$temporary/bundle"
for artifact in "${artifacts[@]}"; do
  extension=klib
  [[ $artifact != koblas && $artifact != koblas-jvm ]] || extension=jar
  base="com/eignex/$artifact/$version/$artifact-$version"
  for suffix in ".$extension" .pom .module -sources.jar -javadoc.jar; do
    grep -Fxq "$base$suffix" "$temporary/payloads" || fail "missing artifact: $base$suffix"
  done
  pom="$temporary/bundle/$base.pom"
  xmllint --nonet --noout "$pom"
  for field in groupId artifactId version; do
    case "$field" in groupId) expected=com.eignex ;; artifactId) expected=$artifact ;; version) expected=$version ;; esac
    actual=$(xmllint --nonet --xpath "string(/*[namespace-uri()='http://maven.apache.org/POM/4.0.0']/*[local-name()='$field' and namespace-uri()='http://maven.apache.org/POM/4.0.0'])" "$pom")
    [[ $actual == "$expected" ]] || fail "wrong POM $field: $artifact"
  done
  for field in name description url licenses/license/name licenses/license/url developers/developer/id scm/url scm/connection; do
    xpath='/*'
    IFS=/ read -r -a parts <<<"$field"
    for part in "${parts[@]}"; do xpath+="/*[local-name()='$part' and namespace-uri()='http://maven.apache.org/POM/4.0.0']"; done
    actual=$(xmllint --nonet --xpath "string($xpath)" "$pom")
    [[ -n $actual ]] || fail "missing POM $field: $artifact"
  done
  jq -e --arg version "$version" --arg artifact "$artifact" '
    .component | .group == "com.eignex" and .module == "koblas" and .version == $version and
      ($artifact == "koblas" or .url == "../../koblas/\($version)/koblas-\($version).module")
  ' "$temporary/bundle/$base.module" >/dev/null || fail "wrong module coordinates or root reference: $artifact"
done
while IFS= read -r name; do
  grep -Fxq "$name.asc" "$temporary/files" || fail "missing signature: $name"
done <"$temporary/payloads"
while IFS= read -r name; do
  case "$name" in
    *.asc) grep -Fxq "${name%.asc}" "$temporary/payloads" || fail "orphan signature: $name" ;;
    *.md5|*.sha1|*.sha256|*.sha512)
      target=${name%.*}
      grep -Fxq "$target" "$temporary/files" || fail "orphan checksum: $name"
      algorithm=${name##*.}
      actual=$(openssl dgst "-$algorithm" "$temporary/bundle/$target")
      expected=$(tr -d '[:space:]' <"$temporary/bundle/$name")
      [[ ${actual##* } == "$expected" ]] || fail "invalid checksum: $name"
      ;;
  esac
done <"$temporary/files"
[[ -n ${SIGNING_KEY:-} ]] || fail 'SIGNING_KEY is required to verify the bundle'
mkdir -m 700 "$temporary/gnupg"
printf '%s\n' "$SIGNING_KEY" | gpg --homedir "$temporary/gnupg" --batch --import >/dev/null 2>&1 || fail 'could not import the release signing key'
while IFS= read -r name; do
  gpg --homedir "$temporary/gnupg" --batch --verify "$temporary/bundle/$name.asc" "$temporary/bundle/$name" \
    >/dev/null 2>&1 || fail "invalid signature: $name"
done <"$temporary/payloads"
echo "Verified $(wc -l <"$temporary/payloads" | tr -d ' ') signed artifacts across ${#artifacts[@]} publications at $version"
