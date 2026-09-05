# Shared prologue for the native build scripts. Sourced, never executed.
#
# Every native build takes the same shape: parse --platform and --output, read a pinned version and its
# checksum out of a .lock file, fetch the archive into a shared download cache, and refuse a platform this
# host has no toolchain for. Only what comes after differs per library.

# Parses the arguments every native build script takes into `platform`, `output` and `blas`.
#
# [usage] is printed on anything unexpected. Set `accepts_blas=1` before calling to take --blas as well,
# which then becomes required; a script that does not set it rejects --blas as it always has.
parse_native_build_args() {
    local usage="$1"; shift
    platform=""; output=""; blas=""
    while (($#)); do
        case "$1" in
            --platform) platform="${2:-}"; shift 2 ;;
            --output) output="${2:-}"; shift 2 ;;
            --blas)
                [[ "${accepts_blas:-0}" == 1 ]] || { echo "$usage" >&2; exit 2; }
                blas="${2:-}"; shift 2 ;;
            *) echo "$usage" >&2; exit 2 ;;
        esac
    done
    [[ -n "$platform" && -n "$output" ]] || { echo "$usage" >&2; exit 2; }
    [[ "${accepts_blas:-0}" != 1 || -n "$blas" ]] || { echo "$usage" >&2; exit 2; }
}

# Reads the pinned `version`, `url` and `expected` checksum out of the lock file [lock].
read_native_lock() {
    local lock="$1"
    version="$(sed -n 's/^version=//p' "$lock")"
    url="$(sed -n 's/^url=//p' "$lock")"
    expected="$(sed -n 's/^sha256=//p' "$lock")"
}

# Downloads [url] to [cache] unless it is already there, then checks it against [expected]. [what] names the
# library in the failure, since a stale cache entry is what a mismatch usually means.
fetch_verified_archive() {
    local url="$1" cache="$2" expected="$3" what="$4"
    mkdir -p "$(dirname "$cache")"
    if [[ ! -f "$cache" ]]; then curl --fail --location --silent --show-error "$url" -o "$cache"; fi
    local actual
    actual="$(sha256sum "$cache" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || { echo "$what source checksum mismatch" >&2; exit 1; }
}

# Refuses a [platform] this host cannot build, since these builds use the host's own toolchain rather than
# cross-compiling. [what] names the library in the failure.
require_platform_host() {
    local platform="$1" what="$2"
    case "$platform" in
        linux-x86_64|linux-arm64)
            [[ "$(uname -s)" == Linux ]] || { echo "$platform requires Linux" >&2; exit 1; } ;;
        macosx-arm64)
            [[ "$(uname -s)" == Darwin ]] || { echo "$platform requires macOS" >&2; exit 1; } ;;
        *) echo "unsupported $what platform $platform" >&2; exit 1 ;;
    esac
}
