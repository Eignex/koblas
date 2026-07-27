<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

# Koblas

[![Maven Central](https://img.shields.io/maven-central/v/com.eignex/koblas.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/com.eignex/koblas)
[![Build](https://github.com/eignex/koblas/actions/workflows/build.yml/badge.svg)](https://github.com/eignex/koblas/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/eignex/koblas/branch/main/graph/badge.svg)](https://codecov.io/gh/eignex/koblas)
[![License](https://img.shields.io/github/license/eignex/koblas)](https://github.com/eignex/koblas/blob/main/LICENSE)

Koblas provides dense and sparse linear algebra for Kotlin Multiplatform: a
well-defined subset of double-precision BLAS and LAPACK, sparse LU
factorization, and a swappable compute backend with optional OpenBLAS-backed
implementations for the JVM and native targets.

## Overview

Kotlin has no standard multiplatform linear algebra library. Koblas is a small
one built around serializable containers: DenseMatrix (flat, row-major),
DenseVector and SparseVector behind a shared view contract, and a CSC
SparseMatrix. Arithmetic lives as free functions over the views; the heavier
dense operations sit behind the swappable LinearAlgebra interface.

The operation set is what a revised simplex or Bayesian-filtering workload
needs: level 1 through 3 BLAS kernels, the LU, Cholesky, QR, and symmetric
indefinite solver families with condition estimation, and sparse basis
factorization with rank-one updates. See [BLAS Coverage](#blas-coverage) for
the exact contract and what is out of scope.

### Installation

```kotlin
implementation("com.eignex:koblas:<version>")
```

The serializable containers need the kotlinx.serialization runtime; there are
no other dependencies.

## Usage

Solve a general dense system via LU:

```kotlin
val rows = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
val a = DenseMatrix.of(rows)
val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
```

Solve a symmetric positive-definite system via Cholesky, then track a rank-one
change without refactorizing:

```kotlin
val l = a.cholesky() // A = L·Lᵀ
val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))
// The factor now tracks A + v·vᵀ.
l.choleskyUpdateInPlace(DenseVector.of(doubleArrayOf(0.5, 1.0)))
```

Factorize a sparse basis and solve both directions:

```kotlin
val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
val s = SparseMatrix.ofColumns(2, 2, cols)
val lu = SparseLu.factorize(s)!!
val forward = lu.ftran(doubleArrayOf(3.0, 5.0)) // B x = b
val backward = lu.btran(doubleArrayOf(3.0, 5.0)) // Bᵀ x = b
```

---

## BLAS Coverage

Every routine in this table is implemented in portable Kotlin and tested
against reference results on every target.

| Standard routine | Koblas |
|------------------|--------|
| ddot, daxpy, dscal | dot, axpy, scale (sparse-aware) |
| dnrm2, dasum, idamax | norm2, asum, iamax |
| dcopy, dswap | copy, swap |
| dgemv (full alpha/beta form) | LinearAlgebra.gemv |
| dger (rank-one update) | addOuter |
| dsymv, dsymm (symmetric multiply) | LinearAlgebra.symv, symm |
| dtrsv, dtrsm (triangular solves) | trsv, trsm |
| dtrmv, dtrmm (triangular multiply) | trmv, trmm |
| dgemm (full form, transpose flags) | LinearAlgebra.gemm |
| dsyrk (symmetric rank-k update) | LinearAlgebra.syrk |
| dgetrf, dgetrs (LU) | factor, solve, plus determinant |
| dgecon, dlange (condition estimate) | LinearAlgebra.rcond, norm1 |
| dpotrf, dpotrs, dpotri (Cholesky) | cholesky, solveSpd, invertSpd |
| dgeqrf, dormqr, dgels (QR) | LinearAlgebra.qr, applyQ, solveLeastSquares |
| dsytrf, dsytrs (symmetric indefinite LDLᵀ) | LinearAlgebra.ldl, solve |
| Cholesky update/downdate | choleskyUpdateInPlace, choleskyDowndateInPlace |

Deviations from the standard are small and documented on each function:

- syrk, symv, symm, and ldl have no uplo parameter: syrk writes the full
  symmetric matrix, the others read the lower triangle only.
- trsm, trmm, and symm operate from the left only.
- Solves take a single right-hand side; use trsm for blocks.
- Least squares requires at least as many rows as columns and full column
  rank.
- cholesky regularizes non-positive-definite pivots unless asked to be
  strict.
- rcond is an estimate that never understates the conditioning.
- norm2 skips the overflow rescale, so components must stay within roughly
  1e150.

Every alpha/beta form follows the BLAS convention that beta equal to zero
overwrites the output without reading it. Factorizations use the LAPACK
packed formats, so a decomposition from one backend solves correctly on any
other.

**Out of scope:** single precision, complex numbers, banded and packed storage
layouts, right-side trsm and trmm, SVD, and eigendecompositions. Nothing is
supported silently: new routines are implemented, tested, and added to the
table when a workload needs them.

---

## Sparse Linear Algebra

SparseLu factorizes a CSC matrix with Markowitz threshold pivoting, keeping
the factors sparse, and solves the forward (FTRAN) and transposed (BTRAN)
systems in time proportional to the nonzeros of the factors. EtaBasis carries
the factorization across rank-one basis changes with the product form of the
inverse, so an update costs one pass over the basis dimension instead of a
refactorization. Together they are the kernel a sparse simplex builds on.

---

## Backends

There are two performance seams. The level 1 kernels (dot, axpy, scale)
dispatch at compile time: the JVM uses the incubator Vector API when started
with `--add-modules=jdk.incubator.vector`; everything else is scalar.
mathBackend reports which kernel was resolved.

The heavier operations, the level 2 and 3 multiplies and the factorization
families with their solves, sit behind the runtime LinearAlgebra interface.
On the JVM a backend on the classpath activates itself through the service
loader; other targets default to the portable reference implementation and
activate a backend with installLinearAlgebra. Storage is flat, row-major
DoubleArray, so a native backend receives raw buffers with no repacking, and
every backend must match the reference on the conformance suite.

The optional koblas-openblas artifact provides a JVM backend built on OpenBLAS
through the Bytedeco presets, with natives bundled for all major platforms:

```kotlin
runtimeOnly("com.eignex:koblas-openblas:<version>")
```

It speeds up matrix products and dense LU factorization by roughly an order of
magnitude at dimension 1000. OpenBLAS runs single-threaded by default, which
is both the fast and the safe configuration under the JVM; the
koblas.openblas.threads system property opts into its threading. Setting
koblas.backend to reference forces the portable implementation.

On the Linux and macOS native targets the optional koblas-cblas artifact
provides the same operations through the system-installed OpenBLAS. It needs
the libraries at link time (libopenblas-dev and liblapacke-dev on
Debian/Ubuntu, brew install openblas on macOS) and is activated once at
startup:

```kotlin
implementation("com.eignex:koblas-cblas:<version>")
```

```kotlin
installLinearAlgebra(CblasLinearAlgebra())
```

It also keeps OpenBLAS single-threaded by default; set OPENBLAS_NUM_THREADS
to opt into its threading.
