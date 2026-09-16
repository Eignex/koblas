#!/usr/bin/env bash
# Install BLAS packages from a directory of cached .deb files, downloading them only on a cache miss.
#
# usage: install-host-blas.sh <cache-dir> <package>...
#
# The bindings dlopen by soname, so the runtime libraries are what matter. Packages are cached as .deb files
# rather than as an unpacked tree, so a hit installs with no mirror access at all and dpkg still works out the
# unpack and configure order itself.
set -eu

cache=${1:?cache directory is required}
shift
packages="$*"
[ -n "$packages" ] || { echo "::error::no packages given"; exit 1; }

# The runner's mirrorlist leads with azure.archive.ubuntu.com, which has been unreachable: apt waits on it for
# every index file, logs Ign, and only then falls back to archive.ubuntu.com, which answers fine. That
# accumulated per-index wait is the difference between an 11 second install and a 20 minute one. Pointing at
# the mirror that answers skips it, and a short acquire timeout keeps a future dead host from being waited on
# rather than abandoned.
mirrors=/etc/apt/apt-mirrors.txt
if [ -f "$mirrors" ]; then
  sudo sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' "$mirrors"
  echo "mirrorlist now:"; cat "$mirrors"
fi

# Generous on purpose: a slow mirror should cost time, never a red build. Observed downloads run 11 to 18
# seconds normally, 117 to 156 seconds on a degraded mirror that still works, and sometimes hang indefinitely.
# Two earlier passes at this were cut too fine, 180s for the step and then 100s per attempt, and both turned a
# working-but-slow install into a failure. The per-attempt cut is still what makes the retry mean anything,
# since bounding only the step lets one hung attempt consume the lot.
#
# Only the download is bounded and retried. Unpacking is local and cannot hang on the network, and on a cache
# hit nothing is downloaded at all.
opts="-o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 -o Acquire::Retries=2"
if [ -z "$(ls -A "$cache" 2>/dev/null || true)" ]; then
  mkdir -p "$cache"
  # Cleared first so the cache holds this closure and not whatever the image had lying around.
  sudo apt-get clean
  for attempt in 1 2; do
    if timeout 600 sudo apt-get $opts update && \
       timeout 600 sudo apt-get $opts install -y --no-install-recommends --download-only $packages; then
      sudo cp /var/cache/apt/archives/*.deb "$cache"/
      sudo chown -R "$(id -u):$(id -g)" "$cache"
      break
    fi
    echo "attempt $attempt failed"
    # A killed apt can leave dpkg mid-transaction, which would fail the next attempt for an unrelated reason.
    sudo dpkg --configure -a || true
    sleep 5
  done
else
  echo "using $(ls -1 "$cache"/*.deb | wc -l) cached packages"
fi

if [ -z "$(ls -A "$cache" 2>/dev/null || true)" ]; then
  echo "::error::could not download $packages; the package mirror is likely unreachable"
  exit 1
fi

# One dpkg call over the whole closure, so it can work out the unpack and configure order itself.
sudo dpkg -i "$cache"/*.deb
