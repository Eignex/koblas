# Usage

If you use NumPy or PyTorch, start with the same idea: express work as vector and matrix operations
rather than loops over individual elements. Koblas provides those operations for double-precision
vectors and matrices backed by ordinary Kotlin arrays. BLAS is the standard set of numerical routines
underneath this API; you do not need to know their names to get started.

See the [README](README.md) to add the dependency. The [engine guide](BACKENDS.md) explains optional
acceleration; the examples here work without a native library.

## Vectors and matrices

Create a matrix from rows and a vector from a one-dimensional array:

```kotlin
import com.eignex.koblas.*

val a = DenseMatrix.ofRows(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))
val b = DenseMatrix.diagonal(2) // Identity matrix

val y = a * x
val product = a * b
val scaled = 2.0 * x
val squaredLength = x dot x
```

Here, matrix `*` means matrix multiplication, like `a @ b` in NumPy or PyTorch. Multiplying by a scalar
scales each element, and `+` and `-` add or subtract matching shapes. These arithmetic operators create
new vectors or matrices; a dot product returns a scalar. Access individual elements with
`a[row, column]` or `x[index]`.

Dense containers store every element. Sparse containers store only selected positions and are useful
when most entries are absent. Matrix products accept either kind: two sparse matrices produce a sparse
result, while a product with a dense matrix produces a dense result. Sparse inputs stay sparse during
the computation. See the [API overview](koblas/module.md) for construction and the full operation set.

## Reusing arrays

For repeated work, write into an existing destination instead of allocating a result each time.
The names come from BLAS: GEMV means general matrix-vector multiplication, and GEMM means general
matrix-matrix multiplication.

```kotlin
val destination = DoubleArray(a.rows)
a.gemvInto(x, destination) // Same calculation as a * x
```

The fuller forms can scale the product and add it to the destination in one call. For matrices,
GEMM computes `c = alpha * a * b + beta * c` when neither operand is transposed:

```kotlin
val result = DenseMatrix.zero(a.rows, b.cols)
val workspace = Workspace()

repeat(1_000) {
    koblas.gemm(
        alpha = 1.0, a = a, transposeA = false,
        b = b, transposeB = false, beta = 0.0, c = result,
        workspace = workspace,
    )
}
```

With beta set to zero, each call replaces the result; setting it to one accumulates into the result.
Transpose flags let you multiply by a transposed operand without creating a separate transposed matrix.
A workspace reuses temporary buffers, such as copies needed when inputs overlap the destination.
It does not remove allocations from expressions that return new results or from sparse operations
that build a new storage pattern. Concurrent calls need separate destinations and workspaces.

You can also reuse storage you already have. The wrapping factories share the supplied array without
copying it. Dense matrix storage is column-major: elements in a column are adjacent, and `a[i, j]`
is stored at `i + j * a.rows`. Use `ofRows` when starting with rows; use `wrap` when your flat array
already has this layout.

A view borrows a rectangular part of a matrix without copying, much like an array slice:

```kotlin
val firstColumn = a.view(columnOffset = 0, cols = 1)
firstColumn[0, 0] = 4.0 // Also changes a[0, 0]
```

The engine accepts views directly for GEMM, GEMV, SYRK, TRSM and TRSV; other matrix APIs may copy them.
A view itself shares storage, but an operation on it can still need temporary buffers. Built-in calls
handle overlapping inputs and destinations by taking snapshots where needed. Direct vendor bindings
have their own overlap rules.

Operations follow the usual BLAS numerical contracts. Different engines can round results differently;
bit-for-bit agreement is not promised. Shapes and arguments are validated, but routines do not scan
matrices for singularity or poor conditioning. Koblas supplies BLAS operations; factorizations and
general solvers belong in the calling library.

## Java and mobile projects

From Java, factories are static methods and Kotlin operators and extensions are exposed through Koblas:

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

Android uses the same dependency as desktop JVM projects, with Java 17 bytecode support in your Android
Gradle Plugin and a minimum API level of 24. It uses the portable implementation; no NDK or native
library setup is needed.

For iOS, add the dependency to commonMain and declare the device and simulator targets:

```kotlin
kotlin {
    iosArm64()
    iosSimulatorArm64()
    sourceSets.commonMain.dependencies {
        implementation("com.eignex:koblas:<version>")
    }
}
```

Linking frameworks and running simulator tests require macOS with Xcode. No numerical library needs installing.
