#!/usr/bin/env bash
# Exercise release version selection and the bundle accepted by the upload step.
set -euo pipefail
scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$scripts/archive-helpers.sh"
temporary=$(mktemp -d)
cleanup() {
  gpgconf --homedir "$temporary/gnupg" --kill all 2>/dev/null || true
  rm -rf "$temporary"
}
trap cleanup EXIT
fail() { echo "$*" >&2; exit 1; }
reject() {
  local expected=$1
  shift
  if "$@" >"$temporary/rejected.log" 2>&1; then fail "Expected rejection: $expected"; fi
  grep -Fq "$expected" "$temporary/rejected.log" || { cat "$temporary/rejected.log" >&2; fail "Missing rejection: $expected"; }
}
extract() {
  bash -c 'source "$1"; extract_verified_archive "$2" "$3" "$4"' bash "$scripts/archive-helpers.sh" "$@"
}
mkdir -p "$temporary/artifact/nested"
echo publication >"$temporary/artifact/nested/publication.pom"
(cd "$temporary/artifact" && zip -qr "$temporary/artifact.zip" .)
digest="sha256:$(sha256 "$temporary/artifact.zip")"
extract "$temporary/artifact.zip" "$digest" "$temporary/extracted"
cmp "$temporary/artifact/nested/publication.pom" "$temporary/extracted/nested/publication.pom"
reject 'digest mismatch' extract "$temporary/artifact.zip" sha256:wrong "$temporary/corrupt"
[[ ! -e $temporary/corrupt ]]
reject 'no SHA-256 digest' extract "$temporary/artifact.zip" '' "$temporary/missing-digest"
echo publication >"$temporary/artifact/outside.pom"
(cd "$temporary/artifact/nested" && zip -q "$temporary/unsafe.zip" ../outside.pom)
reject 'unsafe artifact path' extract "$temporary/unsafe.zip" "sha256:$(sha256 "$temporary/unsafe.zip")" "$temporary/unsafe"
ln -s "$temporary" "$temporary/artifact/link"
(cd "$temporary/artifact" && zip -qy "$temporary/symlink.zip" link)
reject 'unsafe artifact symlink' extract "$temporary/symlink.zip" "sha256:$(sha256 "$temporary/symlink.zip")" "$temporary/symlink"

mkdir "$temporary/repository"
git -C "$temporary/repository" init -q
git -C "$temporary/repository" -c user.name=CI -c user.email=ci@example.invalid commit --allow-empty -qm 'test: create version fixture'
git -C "$temporary/repository" tag v0.1.0
git -C "$temporary/repository" tag v9.0.0-rc.1
resolve() {
  : >"$temporary/outputs"
  (cd "$temporary/repository" && GITHUB_OUTPUT="$temporary/outputs" GITHUB_REF_TYPE=$1 GITHUB_REF_NAME=$2 \
    bash "$scripts/resolve-release-version.sh")
}
resolve branch main
grep -Fxq version=0.1.1-SNAPSHOT "$temporary/outputs"
grep -Fxq is_snapshot=true "$temporary/outputs"
grep -Fxq test_release=false "$temporary/outputs"
grep -Fxq "revision=$(git -C "$temporary/repository" rev-parse HEAD)" "$temporary/outputs"
resolve tag v0.2.0-rc.1
grep -Fxq version=0.2.0-rc.1 "$temporary/outputs"
grep -Fxq is_snapshot=false "$temporary/outputs"
grep -Fxq test_release=true "$temporary/outputs"
resolve tag v0.2.0
grep -Fxq version=0.2.0 "$temporary/outputs"
grep -Fxq is_snapshot=false "$temporary/outputs"
grep -Fxq test_release=false "$temporary/outputs"
git -C "$temporary/repository" tag v1.9.0
git -C "$temporary/repository" tag v1.10.0
resolve branch main
grep -Fxq version=1.10.1-SNAPSHOT "$temporary/outputs"
git -C "$temporary/repository" tag -d v0.1.0 v1.9.0 v1.10.0 >/dev/null
resolve branch main
grep -Fxq version=0.0.1-SNAPSHOT "$temporary/outputs"
for tag in vnext v01.2.0 v0.2.0-rc.01 v0.2.0-SNAPSHOT; do
  reject 'Use a vX.Y.Z or vX.Y.Z-rc.N release tag' resolve tag "$tag"
done
reject 'Releases require a version tag' resolve branch feature

mkdir -m 700 "$temporary/gnupg"
gpg_args=(gpg --homedir "$temporary/gnupg" --batch --pinentry-mode loopback --passphrase '')
"${gpg_args[@]}" --quick-generate-key 'CI <ci@example.invalid>' rsa2048 sign 0 >"$temporary/key.log" 2>&1
SIGNING_KEY=$("${gpg_args[@]}" --armor --export-secret-keys)
export SIGNING_KEY
original="$temporary/original"
artifacts=(koblas koblas-jvm koblas-linuxx64 koblas-linuxarm64 koblas-mingwx64 koblas-macosarm64 koblas-iosarm64 koblas-iossimulatorarm64)
for artifact in "${artifacts[@]}"; do
  directory="$original/com/eignex/$artifact/0.1.1"
  mkdir -p "$directory"
  base="$directory/$artifact-0.1.1"
  cat >"$base.pom" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <groupId>com.eignex</groupId><artifactId>$artifact</artifactId><version>0.1.1</version>
  <name>Koblas</name><description>BLAS</description><url>https://example.invalid</url>
  <licenses><license><name>Apache-2.0</name><url>https://example.invalid/license</url></license></licenses>
  <developers><developer><id>ci</id></developer></developers>
  <scm><url>https://example.invalid</url><connection>scm:git:https://example.invalid</connection></scm>
</project>
EOF
  jq -n --arg artifact "$artifact" '{component:({group:"com.eignex",module:"koblas",version:"0.1.1"} +
    if $artifact == "koblas" then {} else {url:"../../koblas/0.1.1/koblas-0.1.1.module"} end)}' >"$base.module"
  extension=klib
  [[ $artifact != koblas && $artifact != koblas-jvm ]] || extension=jar
  echo binary >"$base.$extension"
  echo sources >"$base-sources.jar"
  echo docs >"$base-javadoc.jar"
  if [[ $artifact == koblas ]]; then echo '{}' >"$base-kotlin-tooling-metadata.json"; fi
  if [[ $artifact == koblas-macosarm64 ]]; then echo metadata >"$base-metadata.jar"; fi
  for payload in "$directory"/*; do "${gpg_args[@]}" --armor --detach-sign "$payload"; done
done
fixture="$temporary/fixture"
reset_fixture() { rm -rf "$fixture"; cp -R "$original" "$fixture"; }
bundle() {
  rm -f "$temporary/bundle.zip"
  (cd "$fixture" && zip -qr "$temporary/bundle.zip" .)
  bash "$scripts/check-release-bundle.sh" "$temporary/bundle.zip" "${1:-0.1.1}"
}
reset_fixture
bundle
mkdir -p "$fixture/com/eignex/koblas-bench/0.1.1"
echo benchmark >"$fixture/com/eignex/koblas-bench/0.1.1/koblas-bench-0.1.1.pom"
reject 'unexpected publication' bundle
reset_fixture
rm "$fixture/com/eignex/koblas-macosarm64/0.1.1/koblas-macosarm64-0.1.1.klib"
reject 'missing artifact' bundle
reset_fixture
reject 'wrong publication version' bundle 0.1.2
base="$fixture/com/eignex/koblas-jvm/0.1.1/koblas-jvm-0.1.1"
sed 's|<description>BLAS</description>||' "$base.pom" >"$temporary/pom"
mv "$temporary/pom" "$base.pom"
reject 'missing POM description' bundle
reset_fixture
rm "$base.jar.asc"
reject 'missing signature' bundle
reset_fixture
echo corrupted >"$base.jar"
reject 'invalid signature' bundle
reset_fixture
echo wrong >"$base.jar.sha256"
reject 'invalid checksum' bundle
reset_fixture
sha256 "$base.jar" >"$base.jar.sha256"
bundle
echo 'CI script checks passed'
