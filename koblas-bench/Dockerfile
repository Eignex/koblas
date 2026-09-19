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

RUN if [ "$(dpkg --print-architecture)" != amd64 ]; then echo "no oneMKL for $(dpkg --print-architecture); stage left as is"; else true \
    && apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && curl --location --silent --show-error --output /tmp/mkl.deb \
        "https://apt.repos.intel.com/oneapi/pool/main/${MKL_DEB}" \
    && echo "${MKL_SHA256}  /tmp/mkl.deb" | sha256sum --check --status \
    && dpkg-deb --extract /tmp/mkl.deb / \
    && ln --symbolic --no-target-directory /opt/intel/oneapi/mkl/2026.1 /opt/intel/oneapi/mkl/latest \
    && test -f /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so.3 \
    && rm /tmp/mkl.deb \
    && rm -rf /var/lib/apt/lists/*; fi

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "onemkl"]

# AOCL, which production selects ahead of everything else on an AMD host, so a fleet without it has never
# measured the library a process on that hardware reaches first.
#
# AMD's own build is behind a licence that does not permit redistribution. The distribution's BLIS is the
# other packaging of the same library and is the candidate Koblas already names: same `libblis.so.4` soname,
# the CBLAS entry points compiled in, and the serial flavour, so one compute thread is what the binary is
# rather than what a call configures it to be. That is a real AOCL route and not AMD's tuning of it, which is
# why the resolved file in the report is the thing to read before comparing a number here with one from a
# machine that installed AMD's package.
FROM onemkl AS aocl

ARG BLIS_VERSION=1.1-3build1

RUN if [ "$(dpkg --print-architecture)" != amd64 ]; then echo "no AOCL route for $(dpkg --print-architecture); stage left as is"; else true \
    && apt-get update \
    && apt-get install --yes --no-install-recommends "libblis4-serial=${BLIS_VERSION}" \
    && test -e /usr/lib/x86_64-linux-gnu/libblis.so.4 \
    && rm -rf /var/lib/apt/lists/*; fi

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "openblas,aocl"]

FROM aocl AS all

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "all"]

# Arm Performance Libraries, the tuned library Koblas prefers on ARM64. OpenBLAS is what selection falls back
# to where none is installed, so a capture without this one measures only the last resort.
#
# Building this stage downloads Arm Performance Libraries, which means accepting Arm's licence for it. That is
# what the licence permits — installing it on a machine you are using. It does not permit redistributing it,
# so this image must not be published anywhere.
#
# Arm ships one self-extracting installer per compiler inside a tar. Running it would accept the licence
# non-interactively and write into system directories, so the payload after its marker line, an ordinary
# gzipped tar, is carved out instead and the package extracted from that.
FROM openblas AS armpl

ARG ARMPL_VERSION=25.07
ARG ARMPL_SHA256=28a0cdf84b1f8e61d1d1ea484f4e2ecf645f7d916bb002ed34d46f6eb2e41345

RUN if [ "$(dpkg --print-architecture)" != arm64 ]; then echo "no ArmPL for $(dpkg --print-architecture); stage left as is"; else true \
    && apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && curl --location --silent --show-error --output /tmp/armpl.tar \
        "https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_${ARMPL_VERSION}/arm-performance-libraries_${ARMPL_VERSION}_deb_gcc.tar" \
    && echo "${ARMPL_SHA256}  /tmp/armpl.tar" | sha256sum --check --status \
    && tar --extract --file /tmp/armpl.tar --directory /tmp \
        "./arm-performance-libraries_${ARMPL_VERSION}_deb/arm-performance-libraries_${ARMPL_VERSION}_deb.sh" \
    && installer="/tmp/arm-performance-libraries_${ARMPL_VERSION}_deb/arm-performance-libraries_${ARMPL_VERSION}_deb.sh" \
    && start=$(awk '/^__START_OF_PAYLOAD__/ { print NR + 1; exit }' "$installer") \
    && tail --lines "+$start" "$installer" | tar --extract --gzip --directory /tmp "armpl_${ARMPL_VERSION}_gcc.deb" \
    && dpkg-deb --extract "/tmp/armpl_${ARMPL_VERSION}_gcc.deb" / \
    && echo "/opt/arm/armpl_${ARMPL_VERSION}_gcc/lib" > /etc/ld.so.conf.d/armpl.conf \
    && ldconfig \
    && test -f "/opt/arm/armpl_${ARMPL_VERSION}_gcc/lib/libarmpl_lp64.so" \
    && rm -rf /tmp/armpl.tar "/tmp/arm-performance-libraries_${ARMPL_VERSION}_deb" /tmp/armpl_*.deb \
    && rm -rf /var/lib/apt/lists/*; fi

ENTRYPOINT ["/workspace/koblas-bench/capture-report.sh", "--libraries", "openblas,armpl"]
