# syntax=docker/dockerfile:1

ARG UBUNTU_IMAGE=ubuntu@sha256:513c074113a871b51a8d16ab445c88779d6452d937a164fb5cc479f32668a41d

FROM ${UBUNTU_IMAGE} AS openblas

ARG DEBIAN_FRONTEND=noninteractive
ARG JDK_VERSION=25.0.4+7-1~26.04
ARG OPENBLAS_VERSION=0.3.32+ds-5

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        gcc \
        git \
        libc6-dev \
        "libopenblas-dev=${OPENBLAS_VERSION}" \
        "openjdk-25-jdk-headless=${JDK_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/tmp
WORKDIR /workspace
ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "openblas"]

FROM openblas AS onemkl

ARG INTEL_OPENMP_VERSION=2026.1.2
ARG MKL_VERSION=2026.1.0
ARG ONEMKL_LICENSE_VERSION=2026.1.0
ARG TBB_VERSION=2023.1.0
ARG TCMLIB_VERSION=1.5.0
ARG UMF_VERSION=1.1.0

RUN test "$(dpkg --print-architecture)" = amd64 \
    && apt-get update \
    && apt-get install --yes --no-install-recommends python3 python3-venv \
    && python3 -m venv /opt/onemkl \
    && /opt/onemkl/bin/pip install --no-cache-dir --disable-pip-version-check \
        "intel-openmp==${INTEL_OPENMP_VERSION}" \
        "mkl==${MKL_VERSION}" \
        "onemkl-license==${ONEMKL_LICENSE_VERSION}" \
        "tbb==${TBB_VERSION}" \
        "tcmlib==${TCMLIB_VERSION}" \
        "umf==${UMF_VERSION}" \
    && test -f /opt/onemkl/lib/libmkl_rt.so.3 \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/onemkl/lib
ENV ONEMKL_LIBRARY=/opt/onemkl/lib/libmkl_rt.so.3
ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "onemkl"]

FROM onemkl AS all

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "all"]
