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
one built around serializable containers: DenseMatrix (flat, column-major),
DenseVector and SparseVector behind a shared view contract, and a CSC
SparseMatrix. Arithmetic lives as free functions over the views; the heavier
dense operations sit behind the swappable Blas and Lapack interfaces.

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

Solve a symmetric positive-definite system via Cholesky:

```kotlin
val l = a.cholesky() // A = L·Lᵀ
val xs = solveSpd(l, doubleArrayOf(3.0, 5.0))
```

Factorize a sparse basis and solve both directions:

```kotlin
val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
val s = SparseMatrix.ofColumns(2, 2, cols)
val lu = s.lu()
val forward = lu.solve(doubleArrayOf(3.0, 5.0)) // B x = b
val backward = lu.solve(doubleArrayOf(3.0, 5.0), transpose = true) // Bᵀ x = b
```

---

## BLAS Coverage

Every routine in this table is implemented in portable Kotlin and tested
against reference results on every target.

| Standard routine | Koblas |
|------------------|--------|
| **Level 1** (vector) | |
| ddot, daxpy, dscal | dot, axpy, scale (sparse-aware) |
| dnrm2, dasum, idamax | norm2, asum, iamax |
| dcopy, dswap | copy, swap |
| **Level 2** (matrix-vector, on Blas) | |
| dgemv (full alpha/beta form) | gemv |
| dger (rank-one update) | ger |
| dsymv (symmetric multiply) | symv |
| dtrsv (triangular solve) | trsv |
| dtrmv (triangular multiply) | trmv |
| **Level 3** (matrix-matrix, on Blas) | |
| dgemm (full form, transpose flags) | gemm |
| dsymm (symmetric multiply) | symm |
| dsyrk (symmetric rank-k update) | syrk |
| dtrsm (triangular solve, multi-RHS) | trsm |
| dtrmm (triangular multiply, multi-RHS) | trmm |
| **LAPACK** (factorizations, on Lapack) | |
| dgetrf, dgetrs (LU) | factor, solve; determinant is free |
| dgecon, dlange (condition estimate) | rcond; norm1 is free |
| dpotrf, dpotrs, dpotri (Cholesky) | cholesky, solveSpd, invertSpd |
| dgeqrf, dormqr, dgels (QR) | qr, applyQ, solveLeastSquares, solveMinimumNorm |
| dsytrf, dsytrs (symmetric indefinite LDLᵀ) | ldl, solve |

Semantics follow the standard; the exceptions are documented on each
function. Factorizations use the LAPACK packed formats, so they interchange
between backends.

The API is split by storage. `com.eignex.koblas` holds the containers and the
free-function arithmetic over them; `com.eignex.koblas.dense` holds the dense
seams and routines, `com.eignex.koblas.sparse` the sparse ones. The containers
stay together in the parent package because the view roots are sealed, which is
what gives a serialized snapshot a closed set of concrete storage types.

The dense seam is three interfaces, each named for what it covers: VectorKernels
holds the vector-vector routines, Blas the matrix ones, Lapack the factorizations
built on them, and LinearAlgebra is the Blas and Lapack pair. All three are ranked and selected independently, so a
host providing one library and not the other still accelerates what it can, and
koblas composes the winning halves.

The sparse seam mirrors it one for one -- SparseVectorKernels, SparseBlas,
SparseLapack, SparseLinearAlgebra -- with the same verbs (registerSparseBlas,
installSparseLinearAlgebra), the same priority ranking, and the same fallback to
a portable reference. Knowing one side is knowing the other, and the two
registries are the same file in two packages over one shared Seam type. Where
they genuinely differ is documented at the declaration: the sparse level-1
kernels have no length threshold, and a sparse factorization is an interface
rather than a class because no host solver will describe its factors.

Each group above names the interface its routines belong to, and a member
reaches the installed backend; determinant and norm1 are the two plain functions
here, marked as such. Members also have a free-function spelling of the
same name (trsv, trsm, ger, ...) that forwards to the member, so a call site may
use whichever reads better.

Level 1 is reached differently from the other two, because those kernels do
nanoseconds of work and a virtual call per invocation would cost more than the
kernel. They are specialized at compile time, and consult the VectorKernels interface
only once a run is long enough to cover a foreign call. Three of the level 1
routines stay off that seam on purpose: iamax because its tie-breaking and NaN
ranking are koblas's own contract and idamax implementations disagree about the
latter, copy and swap because copyInto already beats a foreign call.

**Out of scope:** single precision, complex numbers, banded and packed storage
layouts, SVD, and eigendecompositions. Nothing is supported silently: new
routines are implemented, tested, and added to the table when a workload
needs them.

The routines a steady-state loop repeats have a destination-passing form, so a
loop that owns its buffers allocates nothing: solveInto for the dense and
symmetric indefinite solves, single or blocked, applyQInto and the two
least-squares solves for the QR family, solveInto for the sparse
basis and the eta chain, and factorInto to refactorize into existing factor
buffers.

A few routines also need scratch of their own, and those take an optional
Workspace: a pool of vectors keyed by width, which you create and hand over.
Operations borrow and return buffers, so nesting is safe and one workspace
serves whatever dimensions a caller mixes; pools grow on demand, and reserve
pays that cost up front. It is accepted wherever a routine needs temporaries:
rcond and norm1, which are called together to decide whether a factorization
is still accurate enough to reuse, the syrk mirror buffer (an n² scratch, the
largest in the library), ldl, qr, invertSpd, and the blocked multi-RHS solves'
staging. An EtaBasis owns its
scratch
instead, being mutable already, so its solves need no workspace at all.

```kotlin
val ws = Workspace().apply { reserve(n, count = 5) }
val x = DoubleArray(n)
repeat(iterations) {
    basis.solveInto(b, x) // no allocation
    if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
}
```

A Workspace is caller-owned and not thread-safe: give each solver its own.
Passing none keeps the allocating behaviour, which is always correct.

---

## Backends

Backends register themselves and are ranked by priority, one ranking per
interface: whatever the platform provides arrives through registration, and
installLinearAlgebra or installVectorKernels overrides it. On the JVM discovery scans
the classpath; elsewhere it happens at program start.

What differs between the interfaces is not how a backend is selected but when it
is consulted. The level 2 and 3 multiplies and the factorizations amortize a
virtual call, so every invocation goes through the interface. The level 1
kernels do not, so they are specialized at compile time -- the JVM uses the
incubator Vector API when started with `--add-modules=jdk.incubator.vector`, and
the other targets use scalar loops -- and reach a registered VectorKernels backend only
for runs at least as long as the level 1 threshold, 64 elements on the native
targets. Every dense inner loop bottoms out here. The sparse kernels have their
own seam and their own reference, and consult it unconditionally -- there is no
compiled-in sparse primitive for a foreign call to have to beat.

Each threshold has a name and an override: `koblas.dispatch.level1` and its
level2, level3 and lapack counterparts, as a JVM system property or as
`KOBLAS_DISPATCH_LEVEL1` in the environment elsewhere. Storage
is a flat, column-major DoubleArray -- the order LAPACK and Fortran define -- so
a native backend receives raw buffers with no repacking and no row-major wrapper
layer, and every backend must match the reference on the conformance suite.

On the Linux and macOS native targets koblas resolves the host OpenBLAS with
dlopen at program start (libopenblas and liblapacke on Debian/Ubuntu, brew
install openblas on macOS) and uses it for the level 2 and 3 routines, the
factorizations, and level-1 runs long enough to cover a foreign call. Nothing
is bundled and nothing is linked: a host without the libraries runs the
portable kernels, so a shipped binary works either way. It matters most here,
because these targets have no vector kernels: measured on linuxX64 the host
library wins level 2 by 2x to 15x and dense factorization by up to 13x.
OPENBLAS_NUM_THREADS opts into threading; the default is single-threaded.

The JVM resolves the same host OpenBLAS through `java.lang.foreign`, binding it
with `Linker.Option.critical` so a DoubleArray is pinned rather than copied.
Only the routines that win go native: the level-3 products, the LU, LDL and QR
factorizations, the blocked multi-RHS solve, and the condition estimate. The
level-2 routines, the single-vector solves and the whole Cholesky family stay on
the Vector API kernels, which beat a foreign call there — measured, SIMD
Cholesky matches single-threaded OpenBLAS to n=1024 and wins outright at 256.

This is why the JVM artifact targets Java 25: FFM was finalized in 22, and the
backend ships inside koblas rather than as a separate artifact. Native access is
a restricted operation, so pass `--enable-native-access=ALL-UNNAMED` to silence
the warning; without it the backend still works on 25. A machine with no
OpenBLAS runs the portable kernels, as everywhere else.

The ServiceLoader seam remains for a consumer who wants to supply their own
backend, and it outranks the built-in one when its priority is higher.

The seams are independent; print what a runtime resolved with:

```kotlin
println(koblasInfo) // backend=cblas, primitives=scalar+host
```
