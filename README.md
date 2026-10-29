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

Koblas provides dense and sparse double-precision BLAS Levels 1–3 for Kotlin Multiplatform.
Portable Kotlin implementations work without an installed numerical library. JVM performance uses owned
Vector API kernels. Eligible matrix calls can use installed host libraries on JVM and Kotlin/Native.
Koblas implements BLAS; factorizations and general solver workflows belong to consumers.
Koblas is Apache-2.0 licensed. Its container serializers use kotlinx-serialization-core, exposed as an API
dependency. It bundles no vendor libraries or custom native numerical kernels.

## Setup

Add the Maven Central dependency to your JVM or `commonMain` source set:

```kotlin
implementation("com.eignex:koblas:<version>")
```

Supported targets: JVM on JDK 17 or later, Android API 24 or later, Linux x64/arm64, Windows x64, macOS arm64,
and iOS arm64 devices and simulators.

Desktop and Android consume the same JVM Maven artifact. Its multi-release JAR contains portable Java 17
bytecode at the root and the desktop implementations under `META-INF/versions/25`. Android uses the
portable classes; no Android AAR, native binaries, NDK, or native-access flags are needed.

- Desktop JDK 25 or later: pass `--add-modules=jdk.incubator.vector` to enable the SIMD engine. Without it,
  owned kernels use portable Kotlin. With native access enabled, Intel x64 Linux and Windows hosts can also use installed
  oneMKL for Level 3 calls whose rows, columns and depth are all at least 128. Whole-matrix GEMM also
  qualifies with at least 16 output rows and columns and depth from 1024 through 4096. Borrowed views
  retain the larger cutoff. Other calls retain JVM kernels.
- Desktop JDK 17–24 and Android: all operations use portable Kotlin. Host bindings return null.
- Linux, Windows and macOS Kotlin/Native: eligible calls use an installed host library when available, with portable fallback.
- iOS devices and simulators: all operations use portable Kotlin; no vendor library is required.
- Explicit desktop host bindings on JDK 25 or later also need `--enable-native-access=ALL-UNNAMED`. Supported
  vendors include oneMKL, AOCL, Arm Performance Libraries, OpenBLAS and Accelerate.

Windows has no system BLAS. Install an LP64 (32-bit BLAS integers) build of oneMKL, AOCL, ArmPL or OpenBLAS, matching
the process architecture, and put its DLL directory on `PATH` before launching the application. Only
absolute `PATH` entries are searched. Keep runtime dependencies beside the vendor DLL or in Windows
system directories; dependency probing excludes the working directory. Linux and Windows share the preference order: AMD x64 tries AOCL then oneMKL, other
x64 hosts try oneMKL then AOCL, and Arm64 tries ArmPL. OpenBLAS is the final fallback in every case.
Windows CPU identification reads `PROCESSOR_IDENTIFIER` on both runtimes. The bindings retain the ABI
probes and single compute thread requirement, and Windows reuses Linux's automatic routing thresholds.
No DLLs are bundled or downloaded. Windows runtime behavior and thresholds have not been verified on a
Windows host; local validation cross-compiles and links its executables.

Windows candidates include `mkl_rt.3.dll`, `mkl_rt.2.dll`, `mkl_rt.dll`, AOCL's `libblis.dll` and
`libblis-mt.dll` (plus legacy `AOCL-LibBlis-Win-dll.dll` / `AOCL-LibBlis-Win-MT-dll.dll` and unified
`aocl.dll`), `armpl_lp64.dll` / `armpl_lp64_mp.dll`, and `libopenblas.dll` / `openblas.dll`.
ArmPL prefers its serial DLL. A parallel DLL is accepted only when its thread controls can be resolved
and checked; missing controls never count as evidence that a parallel build is serial.
See the [AOCL installation guide](https://docs.amd.com/r/en-US/57404-AOCL-user-guide/4.1.8.-Using-Pre-Built-Binaries)
and [ArmPL Windows guide](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/windows-release-of-arm-performance-libraries)
for their runtime dependencies. Arm64 Windows uses the JVM publication: Kotlin/Native currently offers
only [the Windows x64 target](https://kotlinlang.org/docs/native-target-support.html).

For Android, add the same dependency to your app's `dependencies` block and use a current Android Gradle
Plugin with Java 17 bytecode support. Java 17 bytecode does not require Android 14: D8 desugars supported
language features for the app's minSdk. The library's portable path uses APIs available from API 24.

For iOS, declare the device and simulator targets and put the dependency in `commonMain`:

```kotlin
kotlin {
    iosArm64()
    iosSimulatorArm64()
    sourceSets.commonMain.dependencies {
        implementation("com.eignex:koblas:<version>")
    }
}
```

Linking iOS frameworks and running simulator tests require macOS with Xcode. These targets use Kotlin/Native
publications selected from the same dependency coordinates. No numerical library is bundled or needs installing.

Engine selection is automatic. The [backend guide](BACKENDS.md) explains optional native setup, supported
workloads and measured benefits. Exact engines and route queries are directly available; the
[benchmark guide](koblas-bench/README.md) explains comparison and attribution. See the
[migration guide](MIGRATION.md) when upgrading from the earlier backend and packed-matrix APIs.

Application code can select portable Kotlin for an individual engine without an opt-in:

```kotlin
val engine = KoblasEngine.portable()
val workspace = Workspace()
engine.gemm(1.0, a, false, b, false, 0.0, result, workspace)
```

Keep the engine for repeated calls. It never loads a host library and needs no JVM module or native-access
flags. Container operators and convenience functions continue to use the platform default; call the selected
engine directly to use its kernels.

Arithmetic follows routine-specific BLAS semantics. FMA, reduction order, blocking and backend rounding may
vary. Norms protect finite inputs from avoidable square-sum overflow/underflow; exceptional results and
choices BLAS leaves open follow the selected implementation. Storage ownership and alias safety are separate
guarantees. Concurrent calls need separate destinations and workspaces.

## Kotlin

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

Operators allocate results. BLAS-style and `*Into` calls write into caller-owned destinations, and every
routine that can take scratch accepts a `Workspace` to lend it, whether that scratch is the snapshot of an
operand overlapping the destination, the gather a strided vector needs, or the panels a blocked product
packs:

```kotlin
val result = DenseMatrix.zero(2, 2)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemm(1.0, a, false, b, false, 0.0, result, workspace)
}
```

A workspace retains a bounded set of buffers, and a call that needs none takes nothing from the one it is
handed. It does not eliminate the result a fresh-result overload allocates, nor the arrays a sparse operation
discovering new structure has to build. Concurrent calls need distinct destinations and workspaces.

A rank update can target a rectangle in a reused matrix. For example, a trailing update after pivot `k`
uses the original matrix and vector buffers:

```kotlin
val start = k + 1
val extent = matrix.rows - start
koblas.ger(
    -1.0, multipliers, pivotRow, matrix,
    rowOffset = start, columnOffset = start, rows = extent, columns = extent,
    xOffset = start, yOffset = start,
)
```

The window retains the matrix's leading dimension and leaves other entries untouched. Nonaliasing calls
allocate no views or scratch; aliased vectors are snapshotted over their selected ranges. Windowed GER uses
owned kernels on every platform. Expert diagnostics use `DenseMatrixOperation.GerWindow` and a `DenseCall`
whose rows and columns describe the rectangle.

## Borrowed dense submatrices

`DenseMatrix.view(rowOffset, columnOffset, rows, cols)` borrows a rectangle without copying.
Nested views keep the original buffer and column spacing. Writes through a view change its parent;
`DenseMatrixView.wrap` also accepts an external buffer, offset and leading dimension with unit row stride.
Contiguous `DenseMatrix` factories retain their exact-buffer requirement.

```kotlin
val panel = matrix.view(first, start, height, width)
val trailing = matrix.view(first, first, height, height)
koblas.syrk(-1.0, panel, false, 1.0, trailing, workspace = workspace)
```

`gemm`, `syrk`, `trsm`, `gemv` and `trsv` accept views and contiguous matrices together through
`DenseMatrixRegion`. Portable engines address regions directly and snapshot overlapping inputs;
disjoint regions of the same parent need no snapshot. Product packing and triangular right-hand-side
gathering can still use workspace scratch. On Kotlin/Native, eligible host calls pass each region's offset
and leading dimension to the library in place, snapshotting only an input that overlaps the destination.
A host binding without that addressing packs borrowed regions into contiguous workspace loans and copies
back only destination entries, preserving the unselected triangle of `syrk`.
Other matrix APIs may materialise a view.

For route inspection, set `DenseCall.borrowed` for a call with views, `leadingDimension` for a padded
`trsm` destination and `rightLeadingDimension` for a transposed borrowed right operand in `gemm`.
Factorization algorithms, stabilization policy and their end-to-end benchmarks belong to the consumer.


## Java

Factories are static methods, and Kotlin operators/extensions are exposed through `Koblas`. Common calls
have overloads for default arguments:

```java
import com.eignex.koblas.DenseMatrix;
import com.eignex.koblas.DenseVector;
import com.eignex.koblas.Koblas;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {{2.0, 1.0}, {1.0, 3.0}});
DenseVector x = DenseVector.of(new double[] {3.0, 5.0});
DenseVector y = Koblas.multiply(a, x);

double[] destination = new double[a.getRows()];
Koblas.gemvInto(a, x, destination);
```

Use `Koblas.getDefault()` for the selected engine. Containers expose ordinary primitive arrays when direct
storage access is needed.

## Storage and contracts

- `DenseMatrix` is column-major: `A(i, j)` is stored at `i + j * rows`.
- `SparseMatrix` is validated CSC with ascending row indices. Stored zeros participate in arithmetic;
  absent entries are not evaluated, and cancellation does not implicitly remove stored entries.
- `Matrix` products accept dense, sparse or prepared sparse operands in either position, including independent
  transpose flags. Two sparse operands produce a fresh CSC result; other built-in pairings produce dense results.
  `gemmInto` writes any pairing into a dense destination without densifying sparse inputs.
- Arrays can be wrapped without copying. `StridedVector` and matrix row/column views borrow existing storage.
- Built-in `*Into` paths support aliases through snapshotting. Explicit vendor bindings retain their own
  overlap contract. Symmetric operations read only the selected triangle; triangular solves write in place.
- Prepared sparse matrices are `Matrix` operands: they own their snapshots, run on the engine that prepared
  them, and can be shared by readers with separate mutable scratch.
- Each call uses one compute thread. Defaults are immutable; containers, views and workspaces are mutable.

Shape mismatches throw `DimensionMismatch`; invalid indices throw `IndexOutOfBoundsException`; other invalid
arguments throw `IllegalArgumentException`. BLAS routines perform no singularity or conditioning scan.
Behavior left open by BLAS, such as tie/NaN choices in `idamax`, follows the selected implementation.

## API guide

Most application code needs only `com.eignex.koblas.*`. Raw vector and indexed sparse windows support
consumer algorithms over ordinary arrays; panel geometry and packing layouts are internal.

| Area | API |
|---|---|
| Containers and views | [Matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/Matrix.kt), [sparse matrices](koblas/src/commonMain/kotlin/com/eignex/koblas/SparseMatrix.kt), [vectors](koblas/src/commonMain/kotlin/com/eignex/koblas/Vector.kt) |
| Vector arithmetic | [VectorOps](koblas/src/commonMain/kotlin/com/eignex/koblas/VectorOps.kt) |
| Matrix operations | [MatrixOps](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixOps.kt), [generic products](koblas/src/commonMain/kotlin/com/eignex/koblas/MatrixProducts.kt) |
| BLAS | [DenseBlas](koblas/src/commonMain/kotlin/com/eignex/koblas/dense/DenseBlas.kt), [SparseBlas](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparseBlas.kt) |
| Reuse | [Workspace](koblas/src/commonMain/kotlin/com/eignex/koblas/Workspace.kt), [prepared sparse snapshots](koblas/src/commonMain/kotlin/com/eignex/koblas/PreparedSparseMatrix.kt) |
| Raw solver arithmetic and explicit bindings | [SparsePrimitives](koblas/src/commonMain/kotlin/com/eignex/koblas/sparse/SparsePrimitives.kt), [vendor Blas](koblas/src/commonMain/kotlin/com/eignex/koblas/vendor/Blas.kt) |

See the [API overview](koblas/module.md) and [benchmark guide](koblas-bench/README.md) for kernel contracts,
exact-engine comparisons and reproducible captures.

## Development

Run `./gradlew check lintDocs`. Add `-Pkoblas.noSimd=true` to check the JVM fallback without the Vector API
module; this does not hide installed vendor libraries.

On Windows, `gradlew.bat check lintDocs` runs the same shared tests plus the Windows loader tests.
The full repository gate includes benchmark script tests and needs Git Bash on `PATH` (or an absolute `KOBLAS_BASH` setting).
To require installed DLLs instead of allowing vendor tests to skip, set `KOBLAS_REQUIRE_VENDORS` to a
comma-separated list of `onemkl`, `aocl`, `armpl`, or `openblas`, then run
`gradlew.bat :koblas:jvmTest :koblas:mingwX64Test --rerun-tasks` on x64 with JDK 25.
On Arm64, run `:koblas:jvmTest` with an Arm64 JDK instead. The dense host conformance suite checks every
installed vendor against the scalar reference. Do not set the requirement for Java 17 portable tests,
whose bindings intentionally return null.
