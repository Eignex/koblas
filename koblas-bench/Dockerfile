# syntax=docker/dockerfile:1

# A fixed machine to capture reference reports on, so that a number from a cloud host is comparable with one
# from a laptop: same JDK, same OpenBLAS, same oneMKL, pinned by digest and version rather than by whatever
# the base image happens to carry today.

ARG UBUNTU_IMAGE=ubuntu@sha256:513c074113a871b51a8d16ab445c88779d6452d937a164fb5cc479f32668a41d

FROM ${UBUNTU_IMAGE} AS openblas

ARG DEBIAN_FRONTEND=noninteractive
ARG JDK_VERSION=25.0.4+7-1~26.04
ARG OPENBLAS_VERSION=0.3.32+ds-5

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        git \
        "libopenblas-dev=${OPENBLAS_VERSION}" \
        "openjdk-25-jdk-headless=${JDK_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
WORKDIR /workspace
ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "openblas"]

# oneMKL from Intel's own package, unpacked where Koblas already looks for it, so the capture exercises the
# candidate path a user's install would take rather than one the image arranged. The hash is checked because
# an image that silently built against something else would be an unlabelled arm in every report it produced.
FROM openblas AS onemkl

ARG MKL_DEB=intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb
ARG MKL_SHA256=3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && curl --location --silent --show-error --output /tmp/mkl.deb \
        "https://apt.repos.intel.com/oneapi/pool/main/${MKL_DEB}" \
    && echo "${MKL_SHA256}  /tmp/mkl.deb" | sha256sum --check --status \
    && dpkg-deb --extract /tmp/mkl.deb / \
    && ln --symbolic --no-target-directory /opt/intel/oneapi/mkl/2026.1 /opt/intel/oneapi/mkl/latest \
    && test -f /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3 \
    && rm /tmp/mkl.deb \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "onemkl"]

FROM onemkl AS all

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "all"]
