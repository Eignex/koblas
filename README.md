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
factorization, and a swappable compute backend that uses the host OpenBLAS on
the Linux and macOS native targets when it is installed.

## Overview

Kotlin has no standard multiplatform linear algebra library. Koblas is a small
one built around serializable containers: DenseMatrix (flat, row-major),
DenseVector and SparseVector behind a shared view contract, and a CSC
SparseMatrix. Arithmetic lives as free functions over the views; the heavier
dense operations sit behind the swappable LinearAlgebra interface.

The operation set covers level 1 through 3 BLAS kernels, the LU, Cholesky, QR,
and symmetric indefinite solver families with condition estimation, and sparse
factorization with rank-one basis updates. See [BLAS Coverage](#blas-coverage)
for the exact contract and what is out of scope.

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
| dgeqrf, dormqr, dgels (QR) | qr, applyQ, solveLeastSquares, solveMinimumNorm |
| dsytrf, dsytrs (symmetric indefinite LDLᵀ) | LinearAlgebra.ldl, solve |
| Cholesky update/downdate | choleskyUpdateInPlace, choleskyDowndateInPlace |

Semantics follow the standard; the exceptions are documented on each
function. Factorizations use the LAPACK packed formats, so they interchange
between backends.

**Out of scope:** single precision, complex numbers, banded and packed storage
layouts, SVD, and eigendecompositions. Nothing is supported silently: new
routines are implemented, tested, and added to the table when a workload
needs them.

Every routine that returns a result also has a destination-passing form, so a
loop that owns its buffers allocates nothing: solveInto for the dense and
symmetric indefinite solves, single or blocked, applyQInto and the two
least-squares solves for the QR family, ftranInto and btranInto for the sparse
basis and the eta chain, and factorInto to refactorize into existing factor
buffers.

A few routines also need scratch of their own, and those take an optional
Workspace: a pool of vectors keyed by width, which you create and hand over.
Operations borrow and return buffers, so nesting is safe and one workspace
serves whatever dimensions a caller mixes; pools grow on demand, and reserve
pays that cost up front. It is accepted wherever a routine needs temporaries:
rcond and norm1, which are called together to decide whether a factorization
is still accurate enough to reuse, the syrk mirror buffer (an n² scratch, the
largest in the library), ldl, qr, the Cholesky rank-one update and downdate,
invertSpd, and the triangular solves' staging. An EtaBasis owns its scratch
instead, being mutable already, so its solves need no workspace at all.

```kotlin
val ws = Workspace().apply { reserve(n, count = 5) }
val x = DoubleArray(n)
repeat(iterations) {
    basis.ftranInto(b, x) // no allocation
    if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
}
```

A Workspace is caller-owned and not thread-safe: give each solver its own.
Passing none keeps the allocating behaviour, which is always correct.

---

## Backends

There are two performance seams, split by how much work a call does.

The level 1 kernels (dot, axpy, scale) do nanoseconds of work per call, so
dispatch would cost more than the kernel. They are specialized at compile
time: the JVM uses the incubator Vector API when started with
`--add-modules=jdk.incubator.vector`; everything else is scalar. Every inner
loop runs on this seam, including the sparse kernels.

The level 2 and 3 multiplies and the factorizations amortize dispatch, so
they sit behind the runtime LinearAlgebra interface. Backends activate
themselves: through the classpath on the JVM, by registration at program start
elsewhere. The highest priority wins; installLinearAlgebra overrides. Storage
is flat, row-major DoubleArray, so a native backend receives raw buffers with
no repacking, and every backend must match the reference on the conformance
suite.

On the Linux and macOS native targets koblas resolves the host OpenBLAS with
dlopen at program start (libopenblas and liblapacke on Debian/Ubuntu, brew
install openblas on macOS) and uses it for the level 2 and 3 routines, the
factorizations, and level-1 runs long enough to cover a foreign call. Nothing
is bundled and nothing is linked: a host without the libraries runs the
portable kernels, so a shipped binary works either way. It matters most here,
because these targets have no vector kernels: measured on linuxX64 the host
library wins level 2 by 2x to 15x and dense factorization by up to 13x.
OPENBLAS_NUM_THREADS opts into threading; the default is single-threaded.

The JVM has no native backend and needs no native access. Its portable kernels
are Vector API SIMD, which beat a foreign call at level 1, level 2 and the
single-vector solves outright, and the remaining gap on large dense products
and factorizations was not worth an FFI surface, a JDK floor and a launch flag.
The ServiceLoader seam is still there for a consumer who wants to supply one.

The seams are independent; print what a runtime resolved with:

```kotlin
println(koblasInfo) // backend=cblas, primitives=scalar+host
```
