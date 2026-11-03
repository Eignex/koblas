#!/usr/bin/env bash
set -euo pipefail
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || exit 0

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
directory="$root/koblas-bench/build/vendor-reference"
mkdir -p "$directory"
temporary=$(mktemp -d "$directory/prepare.XXXXXX")
trap 'rm -rf "$temporary"' EXIT
package=intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb
expected=3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93
curl --fail --location --retry 3 --output "$temporary/$package" \
  "https://apt.repos.intel.com/oneapi/pool/main/$package"
printf '%s  %s\n' "$expected" "$temporary/$package" | sha256sum --check -
dpkg-deb --extract "$temporary/$package" "$temporary/extracted"
rm -rf "$directory/lib"
mv "$temporary/extracted/opt/intel/oneapi/mkl/2026.1/lib" "$directory/lib"
# The runtime loads versioned SONAMEs; link aliases are not valid frozen lab inputs.
find "$directory/lib" -type l -delete
printf '%s %s\n' "$package" "$expected" > "$directory/vendor-package.txt"
