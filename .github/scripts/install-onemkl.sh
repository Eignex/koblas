#!/usr/bin/env bash
set -euo pipefail

# Use the same pinned runtime as the reference container, without publishing it with the library.
cache=${1:?cache directory is required}
package=intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb
checksum=3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93
mkdir -p "$cache"
if [[ ! -f "$cache/$package" ]]; then
  curl --fail --location --silent --show-error --retry 3 --max-time 600 \
    "https://apt.repos.intel.com/oneapi/pool/main/$package" --output "$cache/$package"
fi
printf '%s  %s\n' "$checksum" "$cache/$package" | sha256sum --check --status
sudo dpkg-deb --extract "$cache/$package" /
sudo ln --symbolic --force --no-target-directory /opt/intel/oneapi/mkl/2026.1 /opt/intel/oneapi/mkl/latest
test -f /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3
