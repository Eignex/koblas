# Usage

See the [README](README.md) for installation and a first example, the [engine guide](BACKENDS.md) for
SIMD and native setup, and the [API overview](koblas/module.md) for the available operations.

## Android and iOS setup

Desktop and Android consume the same JVM Maven artifact. Its multi-release JAR contains portable Java 17
bytecode at the root and the desktop implementations under `META-INF/versions/25`. Android uses the
portable classes; no Android AAR, native binaries, NDK, or native-access flags are needed.

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

## An explicit portable engine

Application code can select portable Kotlin for an individual engine without an opt-in:

```kotlin
val engine = KoblasEngine.portable()
val workspace = Workspace()
engine.gemm(1.0, a, false, b, false, 0.0, result, workspace)
```

Keep the engine for repeated calls. It never loads a host library and needs no JVM module or native-access
flags. Container operators and convenience functions continue to use the platform default; call the selected
engine directly to use its kernels.

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

Arithmetic follows routine-specific BLAS semantics. FMA, reduction order, blocking and backend rounding may
vary. Norms protect finite inputs from avoidable square-sum overflow/underflow; exceptional results and
choices BLAS leaves open follow the selected implementation. Storage ownership and alias safety are separate
guarantees. Concurrent calls need separate destinations and workspaces.
