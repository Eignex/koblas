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

Koblas provides dense and sparse double-precision linear algebra for Kotlin Multiplatform: BLAS Levels 1 to
3, mutable matrices and vectors, and views into existing storage. It has no external dependencies and is
entirely Apache-2.0 licensed.

Every level is common Kotlin, so nothing here needs an installed numerical library. Dense Level 2 and 3
scheduling is shared and portable, and only the arithmetic inside a window is chosen locally. What a
platform picks on top of that floor is a selection, never a requirement:

- **JVM.** With `jdk.incubator.vector` the default engine is the Vector API kernels this library owns, the
  same engine `BuiltinEngines.simd` names; a window too short or too strided for a vector body falls to the
  portable one. Without the module the default is portable Kotlin throughout. No host library is composed
  into an ordinary call on this runtime.
- **Kotlin/Native.** The default engine adds an installed host library where one is present: Level 1 above
  its measured widths, and a whole dense Level 2 or 3 call once the call carries enough arithmetic to pay
  for reaching it. Everything else stays portable, including an operation the library does not export, a
  product over operands packed for this library's own register tile, and a routine or multiplier whose
  documented result a library need not give.
- **Sparse.** Scheduling is this library's own portable CSC code on both runtimes, because no vendor's
  sparse API is bound here. Only the Level 1 kernel a column reaches can be a library's.

Ask rather than assume which of those ran. `KoblasEngine.explain` names the Level 1 implementation a call
reaches; `KoblasEngine.denseRouteOf` and `SparseBlas.matrixRouteOf` name what a matrix call executes,
including the resolved library and symbol where one was reached. An engine's name is a selection, not
evidence of execution.

Explicit oneMKL, AOCL, Arm Performance Libraries, Accelerate and OpenBLAS bindings are separately callable
for comparison and Native acceleration, each held to one compute thread.

Where the standard leaves something open — which index `idamax` returns on a tie or a NaN, the order a sum
accumulates in — the answer is the selected implementation's, and Koblas says so rather than promising one.

The [benchmark guide](koblas-bench/README.md) times each arm against the others and against the vendor
libraries installed on the measuring host. We welcome benchmark reports from different hardware.

## Setup

Add the Maven Central dependency to your JVM or `commonMain` source set. Replace `<version>` with the version
shown in the badge above:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets are JVM (JDK 25 or later), Linux x64/arm64, and macOS arm64. When the Vector API is made
stable that will be the new JVM lowest target.

Two JVM runtime flags, both optional:

- `--add-modules=jdk.incubator.vector` selects the Vector API engine. Without it the default is portable
  Kotlin, which computes the same results.
- `--enable-native-access=ALL-UNNAMED` is needed only to call a host binding explicitly, because those use
  foreign downcalls. Built-in computation never needs it, and the JVM default resolves no library, so
  whether one is installed or loadable cannot affect it.

Host comparisons need a vendor BLAS installed; Koblas ships and packages none. The binding reports the
resolved binary file, identity and available version and thread evidence from the object that performs the
call. Kotlin/Native needs no flag: the binding is resolved through the dynamic loader at first use, and the
portable schedule runs where it is not. macOS host comparisons use system Accelerate.

## Quick start

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.ofRows(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val b = DenseMatrix.diagonal(2)
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))

val product = a * b
val y = a * x
```

Operators create a new result. BLAS-style and `*Into` operations instead write into arrays the caller already
owns, so a loop allocates nothing of its own:

```kotlin
val input = DoubleArray(a.rows)
val output = DoubleArray(a.cols)

repeat(1_000) {
    koblas.gemv(1.0, a, input, 0.0, output, transpose = true)
}
```

## Java

The JVM artifact exposes container factories as static methods and collects Kotlin's operators and extensions on
the `Koblas` class directly from their canonical declarations. Default arguments have Java overloads, so common
calls do not need `Companion`, placeholder flags, or generated `*Kt` class names:

```java
import com.eignex.koblas.DenseMatrix;
import com.eignex.koblas.DenseVector;
import com.eignex.koblas.Koblas;
import com.eignex.koblas.SparseMatrix;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {
    {2.0, 1.0},
    {1.0, 3.0},
});
DenseVector x = DenseVector.of(new double[] {3.0, 5.0});

DenseVector y = Koblas.multiply(a, x);
double norm = Koblas.norm2(x);

double[] destination = new double[a.getRows()];
Koblas.gemvInto(a, x, destination);

SparseMatrix diagonal = SparseMatrix.ofTriplets(
    2,
    2,
    new int[] {0, 1},
    new int[] {0, 1},
    new double[] {2.0, 3.0}
);
```

`DenseMatrix`, `DenseVector`, `SparseMatrix`, and `SparseVector` provide static `of`, `ofRows`, `ofTriplets`,
`zero`, `diagonal`, and `wrap` factories as appropriate. Values use ordinary Java primitive arrays. Read entries
with `get(...)`, mutate dense storage with `set(...)`, and use `getData()` or `getValues()` when direct mutable
storage access is intentional. `Koblas.getDefault()` exposes the selected engine.

### The selected engine

Operators such as `a * b` use the global, platform-selected `koblas` engine, which is chosen once and immutable.
Pass a `KoblasEngine` explicitly when a caller should be able to supply one:

```kotlin
fun multiply(engine: KoblasEngine, left: DenseMatrix, right: DenseMatrix): DenseMatrix =
    engine.gemm(left, right)

val product = multiply(koblas, a, b)
```

Naming an implementation other than the selected one is for measuring the two against each other, so
`BuiltinEngines` sits behind the `KoblasEngineApi` opt-in. It is not a way to pick a faster arm for a given
size: the SIMD kernels already fall back to the portable ones below their lane width, for any strided run,
and for every elementwise operation.

`KoblasEngine.denseRouteOf(operation, call)` reports a dense matrix call in full: the scheduling that owns
it, the body each window reaches at that window's own length, the grouping the backend recommended, and
whether every unit of work reached the same body or the call is a composition.

## API structure

Most application code needs only `com.eignex.koblas.*`. The root package contains the owning dense and sparse
containers, zero-copy vector views, allocating operators, and high-level operations. `Matrix` and `Vector` are
read-only contracts that custom types can implement. `MatrixStorage` and `VectorStorage` identify Koblas's
built-in dense and sparse containers, so storage-level operations can use one API and dispatch according to the
actual storage. `DenseVector` is sealed over the two dense spacings, adjacent and strided. Generic matrix
products dispatch on runtime storage: dense by dense, dense by sparse, sparse by dense and sparse by sparse all
reach the implementation their storage calls for, with no operand densified to get there. Two sparse operands
give a CSC result and every other pairing a dense one, whatever the static types were.

The `com.eignex.koblas.dense`, `com.eignex.koblas.sparse` and `com.eignex.koblas.vendor` packages are the
lower-level composition layer. `DenseBlas` exposes Koblas's dense BLAS signatures and `Blas` the standard's own
argument shapes underneath them, while the kernel interfaces and sparse primitives support custom algorithms
over caller-owned storage. Ordinary matrix and vector arithmetic does not require imports from these packages.

`KoblasEngine` connects the layers: it binds the Level 1, sparse, panel, product and triangular kernels this
platform prefers to the shared portable dense and sparse Level 2 and 3 scheduling.
Installed host bindings are separately callable and retain their own execution identity. Root-package operators
and extensions use the platform-selected `koblas` engine.

## Operations

| API | Includes |
|-----|----------|
| [Dense matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/Matrix.kt), [sparse matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/SparseMatrix.kt), [vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/Vector.kt), and [strided vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/StridedVector.kt) | Dense and sparse containers, array wrapping, factories, and borrowed views into existing vector storage. |
| [Vector operations](koblas/src/commonMain/kotlin/com/eignex/koblas/VectorOps.kt) | Dense and sparse dot products, sums, norms, scaling, copy, swap, gather, scatter, and [plane rotations](koblas/src/commonMain/kotlin/com/eignex/koblas/Rot.kt). |
| [Matrix helpers](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixOps.kt) | Allocating [operators](koblas/src/commonMain/kotlin/com/eignex/koblas/Operators.kt), matrix-vector products, rank updates, [triangular operations](koblas/src/commonMain/kotlin/com/eignex/koblas/Triangular.kt), [slices](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixSlices.kt), [scaling and masking](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixScaling.kt), and transpose. |
| [Dense BLAS](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/DenseBlas.kt) | Portable general, symmetric, and triangular matrix products and solves, including `gemmt` and `syr2k`. |
| [Sparse BLAS](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseBlas.kt) and [prepared snapshots](koblas/src/commonMain/kotlin/com/eignex/koblas/PreparedSparseMatrix.kt) | Portable CSC matrix-vector and matrix-matrix products on either side, symmetric and rank-k products, triangular multiply and solve, scaled addition, transpose, and immutable snapshots for repeated use. |
| [Vendor binding](koblas/src/commonMain/kotlin/com/eignex/koblas/vendor/Blas.kt) | The CBLAS entry points in the standard's own argument shapes, library identity, single-thread evidence, and the route a concrete call takes. |
| [Sparse kernels](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseKernels.kt) and [primitives](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparsePrimitives.kt) | Allocation-free indexed arithmetic over caller-owned arrays, including accumulation, touched-index handling, diagnostics, and pivot candidates. |
| [Workspace](koblas/src/commonMain/kotlin/com/eignex/koblas/Workspace.kt) | Reusable scratch for the routines that stage, gather or pack, borrowed by exact length with bounded retention. |
| [Engine](koblas/src/commonMain/kotlin/com/eignex/koblas/KoblasEngine.kt) | The selected engine, its route and Level 1 attribution, and the opt-in seam for naming an implementation to measure. |

Sparse support is storage, Level 1, the generic primitives and portable Levels 2 and 3. Structural semantics
are part of the contract: a stored zero participates, an absent position is never evaluated, a fresh result
owns its arrays and its rows ascend within every column. Factorizations are not here; a consumer that needs one
builds it on `SparsePrimitives` and owns its own factors.

## Error handling

Shape mismatches throw `DimensionMismatch`, invalid logical indices throw `IndexOutOfBoundsException`, and other
invalid arguments throw `IllegalArgumentException`. Numerical failures use `KoblasException` subtypes.

## Storage and reuse

All matrices and vectors use `Double` values. `DenseMatrix` uses column-major storage, so `A(i, j)` is at
`i + j * rows`. `SparseMatrix` uses compressed sparse column (CSC) storage with sorted `Int` row indices.
Compatible arrays can be wrapped without copying them.

A `StridedVector` refers to the original storage instead of copying it, and can address part of an array or a
row or column of a matrix. Every dense routine takes one wherever it takes a contiguous vector, because the
spacing is an increment the library is handed rather than a different algorithm:

```kotlin
val storage = DenseMatrix.zero(512, 32)
val firstColumn = storage.column(0)
val everySecondEntry = DenseVector.wrap(DoubleArray(64)).view(offset = 0, size = 32, stride = 2)

firstColumn.scale(2.0)
val alignment = firstColumn dot everySecondEntry
```

Matrices are passed whole. Level 2 and Level 3 take a `DenseMatrix` with its transpose and stored triangle
travelling beside it as arguments, so there is no submatrix type to construct.

Routines that need temporary storage — staging an operand against an alias, gathering a coefficient column,
packing a panel — take an optional `Workspace`. A call given none allocates its own scratch; a call given one
borrows by exact length and hands the buffer back, so repeated calls over the same shapes reuse that scratch
instead of reallocating it. It bounds the scratch, not the whole call: a routine that returns a fresh result
still allocates it, and the allocation-free path is the `*Into` and BLAS-style form writing into storage the
caller owns. A workspace also retains a bounded number of lengths, so sweeping changing shapes does not make
it grow with the history of the program:

```kotlin
val symmetric = DenseMatrix.zero(512, 512)
val block = DenseMatrix.zero(512, 32)
val result = DenseMatrix.zero(512, 32)
val workspace = Workspace()

repeat(1_000) {
    koblas.symm(1.0, symmetric, block, 0.0, result, workspace = workspace)
}
```

Portable `*Into` operations snapshot inputs that share a destination buffer. Explicit vendor bindings retain
the vendor contract and reject undefined overlap. Triangular solves write their destination in place by design.

For symmetric operations, Koblas reads only the triangle you select. Use `symv` or `symm` to treat that
triangle as a full symmetric matrix.

The engine is immutable and safe to share. Each portable invocation uses the calling thread and every vendor
call uses one compute thread, but matrices, vectors, views and workspaces are mutable and a workspace belongs
to one invocation at a time, so concurrent calls need distinct outputs and distinct workspaces.
