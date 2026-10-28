#!/usr/bin/env bash
set -euo pipefail
bench=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$bench/script-helpers.sh"
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
tree="$temporary/source archive"
mkdir -p "$tree/src" "$tree/build" "$tree/nested/.gradle" "$tree/koblas/build/libs" "$tree/koblas-bench/build/classes"
echo source >"$tree/src/a file.kt"
echo ignored >"$tree/build/generated"
echo ignored >"$tree/nested/.gradle/cache"
echo binary >"$tree/koblas/build/libs/kernel.jar"
echo binary >"$tree/koblas-bench/build/classes/Runner.class"
fingerprint "$tree" sources archive >"$temporary/sources"
[[ $(wc -l <"$temporary/sources" | tr -d ' ') == 1 ]]
grep -Fxq "$(sha256 "$tree/src/a file.kt") src/a file.kt" "$temporary/sources"
ln -s 'a file.kt' "$tree/src/linked.kt"
fingerprint "$tree" sources archive >"$temporary/linked"
grep -Fxq "$(sha256 "$tree/src/a file.kt") src/linked.kt" "$temporary/linked"
rm "$tree/src/linked.kt"
echo changed >"$tree/build/generated"
fingerprint "$tree" sources archive >"$temporary/unchanged"
cmp "$temporary/sources" "$temporary/unchanged"
echo changed >"$tree/src/a file.kt"
fingerprint "$tree" sources archive >"$temporary/changed"
if cmp -s "$temporary/sources" "$temporary/changed"; then echo 'Source changes must change the fingerprint' >&2; exit 1; fi
fingerprint "$tree" binaries archive >"$temporary/binaries"
fingerprint "$tree" binaries >"$temporary/default-binaries"
cmp "$temporary/binaries" "$temporary/default-binaries"
[[ $(wc -l <"$temporary/binaries" | tr -d ' ') == 2 ]]
grep -Fxq "$(sha256 "$tree/koblas/build/libs/kernel.jar") koblas/build/libs/kernel.jar" "$temporary/binaries"
git -C "$tree" init -q
git -C "$tree" add src
echo untracked >"$tree/untracked"
fingerprint "$tree" sources git >"$temporary/git"
cmp "$temporary/changed" "$temporary/git"
if [[ $(uname -s) == Linux ]]; then
  cpu=$(first_allowed_cpu)
  [[ $cpu =~ ^[0-9]+$ ]]
  taskset -c "$cpu" true
fi
bash "$bench/capture-report.sh" --help >/dev/null 2>&1
echo 'Benchmark script checks passed'
