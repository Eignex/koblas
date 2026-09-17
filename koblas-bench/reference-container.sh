#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target=all

usage() {
    echo "usage: koblas-bench/reference-container.sh [openblas|onemkl|armpl|all] [capture options]" >&2
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == openblas || "${1:-}" == onemkl || "${1:-}" == armpl || "${1:-}" == all ]]; then
    target="$1"
    shift
fi

command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }
docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null && sudo docker info >/dev/null 2>&1; then
        docker_command=(sudo docker)
    else
        echo "the Docker daemon is unavailable" >&2
        exit 1
    fi
fi

# BuildKit builds only the stages the target depends on. The legacy builder walks every stage above it in
# the file, so an arm64 host asked for the armpl target would build the oneMKL stage first and fail there.
export DOCKER_BUILDKIT=1
image="koblas-vendors:$target"
build_arguments=()
if "${docker_command[@]}" build --help 2>&1 | grep -q -- '--progress'; then
    build_arguments+=(--progress plain)
fi
"${docker_command[@]}" build "${build_arguments[@]}" --target "$target" --tag "$image" \
    --file "$root/koblas-bench/Dockerfile" "$root/koblas-bench"
docker_arguments=(
    --rm
    --init
    --user "$(id -u):$(id -g)"
    --volume "$root:/workspace"
    --workdir /workspace
)
git_common_dir=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)
if [[ "$git_common_dir" != "$root/"* ]]; then
    docker_arguments+=(--volume "$git_common_dir:$git_common_dir")
fi
"${docker_command[@]}" run "${docker_arguments[@]}" "$image" "$@"
