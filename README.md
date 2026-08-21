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
SparseMatrix, transposable in place of a CSR conversion. Arithmetic lives as
free functions over the views; the heavier dense operations sit behind the
swappable Blas and Lapack interfaces.

The operation set covers level 1 through 3 BLAS kernels, the LU, Cholesky, QR,
and symmetric indefinite solver families with condition estimation, and sparse
LU and symmetric factorizations. See [BLAS Coverage](#blas-coverage)
for the exact contract and what is out of scope.

### Element types

Every container, backend half and factorization names its element type, and
double precision is the only one implemented: `F64DenseMatrix`, `F64Blas`,
`F64LuDecomposition`. The unqualified names are aliases for them, so
`DenseMatrix` is `F64DenseMatrix`, and the short spelling used throughout this
README is the double-precision one.

The names are shaped that way so that a second element type is an addition
rather than a reshaping. `F32` is next: `Float` containers and hand-written
level-1 to level-3 kernels, where a SIMD register holds twice the lanes, with
factorizations that promote to double precision and demote on the way out, which
the seam lets a real single-precision LAPACK replace later. `BF16` is planned as
storage only, `Short` bfloat16 accumulating in `Float`, for the products bf16 is
actually used for and no factorizations, whose eight mantissa bits would not
survive pivoting.

What is not per element type stays shared: `Backend`, `Seam` and the priority
ranking, the shape and bounds checks, the sparse symbolic analysis and ordering,
`Uplo`, `CholeskyPolicy`, and the `Workspace` pooling, which lends buffers of
one array type per element type out of one set of reclamation rules. The dispatch
thresholds are per element type, counted in elements rather than bytes, and
double precision owns the unqualified keys: `koblas.dispatch.level3` is its
level-3 crossover, where a later element type reads
`koblas.dispatch.f32.level3`.

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
val chol = a.cholesky() // A = L·Lᵀ; throws if A is not positive-definite
val xs = chol.solve(doubleArrayOf(3.0, 5.0))
```

An estimate that has drifted slightly indefinite can ask for the nearby factor
instead of an exception, which is a decision the caller makes rather than a
default:

```kotlin
val chol = a.cholesky(CholeskyPolicy.Regularize()) // floors non-positive pivots
```

Factorize a sparse basis and solve both directions:

```kotlin
val cols = listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0))
val s = SparseMatrix.ofColumns(2, 2, cols)
val lu = s.lu()
val forward = lu.solve(doubleArrayOf(3.0, 5.0)) // B x = b
val backward = lu.solve(doubleArrayOf(3.0, 5.0), transpose = true) // Bᵀ x = b
```

`ofColumns` is the readable spelling for a matrix written out in place. Data
arriving from elsewhere is usually coordinate triplets instead — Matrix Market,
`scipy.sparse`, Eigen — which `ofTriplets` takes in primitive arrays, in any
order, summing any repeated position:

```kotlin
val s = SparseMatrix.ofTriplets(
    rows = 2,
    cols = 2,
    rowIdx = intArrayOf(0, 1, 0, 1),
    colIdx = intArrayOf(0, 0, 1, 1),
    values = doubleArrayOf(2.0, 1.0, 1.0, 3.0),
)
```

Every factorization is reached the same way on both storages: a verb on the
matrix produces the decomposition, and the solves hang off the decomposition.

```kotlin
a.lu().solve(b)                    a.lu().invert()      a.lu().rcond(norm1(a))
a.ldl().solve(b)                   a.cholesky().solve(b)
a.qr().solveLeastSquares(b)        a.qr().applyQ(y)
a.qrPivoted().solveLeastSquares(b) // reports a numerical rank
```

Each decomposition is its own type, so a solve will not accept the matrix it
came from, or the factor of a different one. All four take a DenseMatrix; a
foreign MatrixLike reaches them by materialising first
(`DenseMatrix.of(a.toArray()).cholesky()`), which is explicit because the copy
is real.

### When it fails

Every failure koblas raises is an `IllegalArgumentException`, so existing
handlers keep working, and each is also one of three types you can tell apart:

| type | means |
|------|-------|
| `DimensionMismatch` | operands do not fit — a call-site bug |
| `SingularMatrix` | the factorization has no inverse; carries the failing `position` |
| `NotPositiveDefinite` | a Cholesky or strict LDLᵀ pivot was not positive; carries `position` and `pivot` |

The distinction that matters is between a shape that was wrong when the call was
made and a matrix that turned out to be numerically unusable. The second kind is
often recoverable, and catching it is how that decision gets made after the fact
rather than predicted with a policy up front:

```kotlin
val chol = try {
    a.cholesky()
} catch (e: NotPositiveDefinite) {   // drifted indefinite at column e.position
    a.cholesky(CholeskyPolicy.Regularize())
}
```

Solving against a singular factorization throws rather than returning
infinities, on both storages. `singular` and `failedAt` are still there for when
a singular basis is an expected outcome rather than a bug.

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
| drotg, drot (Givens rotation) | rotg, rot |
| **Level 2** (matrix-vector, on Blas) | |
| dgemv (full alpha/beta form) | gemv |
| dger (rank-one update) | ger |
| dsyr, dsyr2 (symmetric rank-1/2 update) | syr, syr2 |
| dsymv (symmetric multiply) | symv |
| dtrsv (triangular solve) | trsv |
| dtrmv (triangular multiply) | trmv |
| **Level 3** (matrix-matrix, on Blas) | |
| dgemm (full form, transpose flags) | gemm |
| dsymm (symmetric multiply) | symm |
| dsyrk (symmetric rank-k update) | syrk |
| dsyr2k (symmetric rank-2k update) | syr2k |
| dtrsm (triangular solve, multi-RHS) | trsm |
| dtrmm (triangular multiply, multi-RHS) | trmm |
| **LAPACK** (factorizations, on Lapack) | |
| dgetrf, dgetrs, dgetri (LU) | factor, solve, invert; determinant is free |
| dgecon, dlange (condition estimate, norms) | rcond; norm1, normInf, normFro are free |
| dtrtri (triangular inverse) | trtri |
| dpotrf, dpotrs, dpotri (Cholesky) | cholesky, solve, invert on CholeskyDecomposition |
| dgeqrf, dormqr, dgels (QR) | qr, applyQ, solveLeastSquares, solveMinimumNorm |
| dgeqp3 (QR with column pivoting) | qrPivoted, reporting a numerical rank |
| dsytrf, dsytrs (symmetric indefinite LDLᵀ) | ldl, solve |
| **Sparse** (on SparseBlas and SparseLapack) | |
| sparse matrix-vector product, both directions | gemv |
| sparse triangular solve | trsv |
| unsymmetric sparse LU (Markowitz pivoting) | lu, solve both directions, determinant |
| symmetric sparse LDLᵀ and Cholesky | analyze, ldl, cholesky; the analysis is reusable |
| fill-reducing ordering (minimum degree) | on by default in analyze; SparseOrdering.Natural opts out |

Semantics follow the standard; the exceptions are documented on each
function. Factorizations use the LAPACK packed formats, so they interchange
between backends.

The API is split by storage. `com.eignex.koblas` holds the containers and the
free-function arithmetic over them; `com.eignex.koblas.dense` holds the dense
seams and routines, `com.eignex.koblas.sparse` the sparse ones. The containers
stay together in the parent package because the view roots are sealed, which is
what gives a serialized snapshot a closed set of concrete storage types.

The dense seam is three interfaces, each named for what it covers: VectorKernels
holds the vector-vector routines, Blas the matrix ones, and Lapack the
factorizations built on them, with LinearAlgebra as the Blas and Lapack pair.
All three are ranked and selected independently, so a host providing one library
and not the other still accelerates what it can, and koblas composes the winning
halves.

The sparse seam mirrors it one for one -- SparseVectorKernels, SparseBlas,
SparseLapack, SparseLinearAlgebra -- and shares its machinery rather than
paralleling it: one registerBackend, one priority ranking per half, one registry,
one KoblasContext holding all six. Knowing one side is knowing the other. Where
they genuinely differ is documented at the declaration: the sparse vector kernels
have no length threshold, and a sparse factorization is an interface rather than a
class because no host solver will describe its factors.

Each group above names the interface its routines belong to, and a member
reaches the installed backend; determinant and norm1 are the two plain functions
here, marked as such. Members also have a free-function spelling of the
same name (trsv, trsm, ger, ...) that forwards to the member, so a call site may
use whichever reads better.

The BLAS-shaped routines keep binary choices as Boolean parameters. Name them at
a call site that supplies more than one, so the standard flags stay readable:
`trsm(a, b, lower = true, transpose = false, unitDiag = false, right = false)`.

The everyday arithmetic also has an operator spelling -- `a * b`, `a * x`,
`a + b`, `2.0 * v`, `-v` -- for the call sites where the expression reads better
than the routine. Every one is an alias that forwards to a routine documented
above and allocates its result, so the in-place forms remain what a loop should
call; each operator's KDoc names the one to reach for. An operator never
introduces a kernel of its own, which is what decides the omissions: there is no
sparse scale and no sparse pattern union, so neither has one. Columns and rows
come out as vectors with `a.column(j)` and `a.row(i)`, the first a contiguous
copy and the second a strided gather, which is the storage order showing through.

`matVec` is the product that takes any MatrixLike against any VectorLike,
sparse-aware on both sides; it is named for `matMul` rather than `gemv` so that
`gemv` means the seam member and nothing else. A diagonal is applied as a
scaling rather than built as a matrix -- `scaleRows(a, d)` is `D · A` and
`scaleColumns(a, d)` is `A · D`, the second being the cheap direction under
column-major storage, and the sparse matrix takes the column form only.

Level 1 is reached differently from the other two, because those kernels do
nanoseconds of work and a virtual call per invocation would cost more than the
kernel. They are specialized at compile time, and consult the VectorKernels
interface only once a run is long enough to cover a foreign call. Three of the
level 1 routines stay off that seam on purpose: iamax because its tie-breaking
and NaN ranking are koblas's own contract and idamax implementations disagree
about the latter, copy and swap because copyInto already beats a foreign call.

**Out of scope:** complex numbers, banded and packed storage layouts, SVD, and
eigendecompositions. Single precision is planned rather than omitted; see
[Element types](#element-types). Nothing is supported silently: new routines are
implemented, tested, and added to the table when a workload needs them.

The routines a steady-state loop repeats have a destination-passing form, so a
loop that owns its buffers allocates nothing: solveInto for the dense and
symmetric indefinite solves, single or blocked, applyQInto and the two
least-squares solves for the QR family, solveInto for the sparse
factorization, and factorInto to refactorize into existing factor buffers.

A few routines also need scratch of their own, and those take an optional
Workspace: a pool of vectors keyed by width, which you create and hand over.
Operations borrow and return buffers, so nesting is safe and one workspace
serves whatever dimensions a caller mixes; pools grow on demand, and reserve
pays that cost up front. It is accepted wherever a routine needs temporaries:
rcond, the syrk mirror buffer (an n² scratch, the largest in the library), ldl,
qr, the SPD invert, and the blocked multi-RHS solves' staging. norm1 is the routine
you call alongside rcond to decide whether a factorization is still accurate
enough to reuse, but it takes no workspace and needs none: a column is
contiguous under column-major storage, so each column sum finishes before the
next begins and one accumulator suffices.

```kotlin
val ws = Workspace().apply { reserve(n, count = 5) }
val x = DoubleArray(n)
repeat(iterations) {
    basis.solveInto(b, x, workspace = ws) // no allocation
    if (koblas.rcond(lu, anorm, ws) < threshold) koblas.factorInto(a, lu)
}
```

A Workspace is caller-owned and not thread-safe: give each solver its own.
Passing none keeps the allocating behaviour, which is always correct.

---

## Backends

Backends register themselves through registerBackend, which offers one object as
every half it implements, and are ranked by priority, one ranking per
interface: whatever the platform provides arrives through registration, and
installBackends overrides it with a KoblasContext of your own.

Discovery runs once, on the first read of koblas, on every platform. It resolves
the host libraries this build ships bindings for -- OpenBLAS and SuiteSparse's
UMFPACK -- by soname, and on the JVM it then scans the classpath for third-party
LinearAlgebra providers. A library that is not installed simply is not
registered, and the portable implementation stays in place, so nothing has to be
configured for either case. On the JVM, `-Dkoblas.backend=reference` registers
nothing at all, and any other value registers only the backend whose name
matches.

Selection is global by default but does not have to be. A KoblasContext holds all
six halves -- the three dense and three sparse -- and is itself a backend, so
`context.gemv(...)` works wherever `koblas.gemv(...)` does. koblas.with(blas =
mine) copies the default and replaces one half, which is what tests, benchmarks
and reproducible runs want instead of mutating process state.

What differs between the interfaces is not how a backend is selected but when it
is consulted. The level 2 and 3 multiplies and the factorizations amortize a
virtual call, so every invocation goes through the interface. The level 1
kernels do not, so they are specialized at compile time -- the JVM uses the
incubator Vector API when started with `--add-modules=jdk.incubator.vector`, and
the other targets use scalar loops -- and reach a registered VectorKernels
backend only for runs at least as long as the level 1 threshold, 64 elements on
the native targets. Every dense inner loop bottoms out here. The sparse kernels
have their own seam and their own reference, and consult it unconditionally --
there is no compiled-in sparse primitive for a foreign call to have to beat.

Each threshold has a name and an override: `koblas.dispatch.level1` and its
level2, level3 and lapack counterparts, as a JVM system property or as
`KOBLAS_DISPATCH_LEVEL1` in the environment elsewhere. Storage
is a flat, column-major DoubleArray -- the order LAPACK and Fortran define -- so
a native backend receives raw buffers with no repacking and no row-major wrapper
layer, and every backend must match the reference on the conformance suite.

The sparse factorization has a host backend too: SuiteSparse's UMFPACK, bound on
the JVM through java.lang.foreign and resolved by soname, so it activates on any
machine with SuiteSparse installed and is absent otherwise. A SparseMatrix
crosses to umfpack_di_* with no repacking at all -- koblas's CSC invariant is
UMFPACK's stated precondition, checked against the headers. Nothing is bundled;
without the library koblas's own Markowitz SparseLu keeps the seam.

On the Linux and macOS native targets koblas resolves the host OpenBLAS with
dlopen at program start (libopenblas and liblapacke on Debian/Ubuntu, brew
install openblas on macOS) and uses it for the level 2 and 3 routines, the
factorizations, and level-1 runs long enough to cover a foreign call. Nothing
is bundled and nothing is linked: a host without the libraries runs the
portable kernels, so a shipped binary works either way. It matters most here,
because these targets have no vector kernels: measured on linuxX64 the host
library wins level 2 by 2x to 15x and dense factorization by up to 13x.
OPENBLAS_NUM_THREADS opts into threading; the default is single-threaded.

That fallback is silent, which is the right default and a bad thing to discover
in production. Ask, when it matters:

```kotlin
koblas.requireAccelerated(BackendSlot.F64Blas, BackendSlot.F64Lapack) // throws, naming the fallback
println(koblas.portableSlots)                                   // or just look
```

Accelerated means a registered backend is being consulted. The compiled-in SIMD
kernels do not count, however fast they are -- they are what you get with no host
library, so mathBackend reporting simd is not evidence that one was found. The
slots are named explicitly because which ones can be accelerated depends on the
target and on what koblas ships: there is no host sparse backend yet, so the
three sparse slots are portable everywhere today.

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
println(koblasInfo) // backend=cblas, kernels=scalar+host
```

---

## Migrating

Every container, backend half and factorization now carries its element type,
and the unqualified names are aliases for the double-precision ones, so code
written against `DenseMatrix`, `Blas` or `LuDecomposition` compiles unchanged.
Three things do move:

| was | now |
|-----|-----|
| `ReferenceLinearAlgebra` | `F64ReferenceLinearAlgebra`; a val, so no alias stands in for it |
| `BackendSlot.Blas`, `BackendSlot.SparseLapack` | `BackendSlot.F64Blas`, `BackendSlot.F64SparseLapack` |
| `"type": "DenseMatrix"` in a snapshot | `"type": "F64DenseMatrix"`, the name of the class that wrote it |

The serial names moved with the class names, so a snapshot an earlier version
encoded through `VectorView` or `MatrixView` needs its discriminator rewritten
before this version reads it. One encoded through a concrete serializer carries
no discriminator and is unaffected.
