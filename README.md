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
factorization, and a swappable backend seam for native BLAS or GPU
implementations.

## Overview

Kotlin has no standard multiplatform linear algebra library. Koblas is a small
one built around serializable containers: DenseMatrix (flat, row-major),
DenseVector and SparseVector behind a shared view contract, and a CSC
SparseMatrix. Arithmetic lives as free functions over the views, and the
heavier dense operations sit behind a runtime-swappable LinearAlgebra
interface.

The operation set is the subset a revised simplex or Bayesian-filtering
workload needs: level 1 through 3 BLAS kernels, LU and Cholesky solver
families, and sparse basis factorization with rank-one updates. See
[BLAS Coverage](#blas-coverage) for the exact contract, including what is
deliberately out of scope.

### Installation

```kotlin
implementation("com.eignex:koblas:<version>")
```

For the serializable containers you also need the kotlinx.serialization
runtime; the library itself has no other dependencies.

## Usage

Solve a general dense system via LU:

```kotlin
val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
```

Solve a symmetric positive-definite system via Cholesky, then track a rank-one
change without refactorizing:

```kotlin
val l = a.cholesky() // A = L·Lᵀ
val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))
l.choleskyUpdateInPlace(DenseVector.of(doubleArrayOf(0.5, 1.0))) // now factors A + v·vᵀ
```

Factorize a sparse basis and solve both directions:

```kotlin
val s = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
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
| dgemm (full form, transpose flags) | LinearAlgebra.gemm |
| dsyrk (symmetric rank-k update) | LinearAlgebra.syrk |
| dgetrf, dgetrs (LU) | factor, solve, plus determinant |
| dgecon, dlange (condition estimate) | LinearAlgebra.rcond, norm1 |
| dpotrf, dpotrs, dpotri (Cholesky) | cholesky, solveSpd, invertSpd |
| Cholesky rank-one update/downdate | choleskyUpdateInPlace, choleskyDowndateInPlace |

Deviations from the standard are small and documented on each function: syrk
has no uplo parameter and always produces the full symmetric matrix, trsm
solves from the left only, LU solve takes a single right-hand side (use
trsm for blocks), cholesky regularizes non-positive-definite pivots unless
asked to be strict, and norm2 skips the overflow rescale so components must
stay within roughly 1e150. Every alpha/beta form follows the BLAS convention
that beta equal to zero overwrites the output without reading it.

**Out of scope:** single precision, complex numbers, banded and packed storage
layouts, right-side trsm, QR, SVD, and eigendecompositions. Level 2 routines
with no consumer here, such as symv and trmv, are also out. Nothing is
supported silently: when a workload needs a new routine it gets implemented,
tested, and added to the table above.

---

## Sparse Linear Algebra

SparseLu factorizes a CSC matrix with Markowitz threshold pivoting (bounded
Suhl and Suhl candidate search), keeping the factors sparse. It solves the
forward system (FTRAN) and the transposed system (BTRAN) in time proportional
to the nonzeros of the factors. EtaBasis maintains the factorization across
rank-one basis changes using the product form of the inverse, so each update
costs a single pass over the basis dimension instead of a refactorization.
Together they are the kernel a sparse simplex builds on.

---

## Backends

There are two performance seams. The level 1 kernels (dot, axpy, scale)
dispatch at compile time: the JVM uses the incubator Vector API when started
with `--add-modules=jdk.incubator.vector` and scalar loops otherwise; all
other targets are scalar. mathBackend reports which kernel was resolved.

The heavier operations (gemv, gemm, syrk, LU) sit behind the runtime
LinearAlgebra interface. On the JVM backends are discovered through the
service loader, so adding one to the classpath activates it without code
changes; all other targets use the portable reference implementation. Storage
is flat, contiguous, row-major DoubleArray, so a native backend receives raw
buffers with no repacking, and every backend must match the reference on the
conformance suite.

The optional koblas-openblas artifact provides a JVM backend built on OpenBLAS
through the Bytedeco presets, with natives bundled for all major platforms:

```kotlin
runtimeOnly("com.eignex:koblas-openblas:<version>")
```

It speeds up matrix products and dense LU factorization by roughly an order of
magnitude at dimension 1000. OpenBLAS runs single-threaded by default, which
is both the fast and the safe configuration under the JVM; the koblas.openblas.threads
system property opts into its threading. Setting koblas.backend to reference
forces the portable implementation regardless of what is on the classpath.
