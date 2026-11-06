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
import com.eignex.koblas.engine.*

val a = DenseMatrix.ofRows(arrayOf(
    doubleArrayOf(2.0, 1.0),
    doubleArrayOf(1.0, 3.0),
))
val x = DenseVector.of(doubleArrayOf(3.0, 5.0))
val b = DenseMatrix.diagonal(2) // Identity matrix

val y = DenseVector.zero(a.rows)
a.gemvInto(x, y)
val product = DenseMatrix.zero(a.rows, b.cols)
a.gemmInto(b, product)
val scaled = DenseVector.wrap(x.toDoubleArray()).also { it.scale(2.0) }
val squaredLength = x dot x
val length = x.norm()              // Euclidean norm, order = 2
val absoluteSum = x.norm(order = 1)
val largestMagnitude = x.norm(order = Int.MAX_VALUE)
```

`gemvInto` writes a matrix-vector product into `y`; `gemmInto` writes a matrix product into `product`.
`scale` changes a vector in place, and a dot product returns a scalar. Access individual elements
with `a[row, column]` or `x[index]`.

Dense containers store every element. Sparse containers store only selected positions and are useful
when most entries are absent. Matrix products accept a sparse operand on either side of a dense operand
and produce a dense result. Sparse inputs stay sparse during the computation. Products of two sparse
matrices are unsupported. See the [API overview](koblas/module.md) for construction and the full
operation set.

The dense matrix facade supplies ordinary product and unit-scaling defaults, and takes vectors without
unpacking their backing arrays, offsets and strides:

```kotlin
a.gemvInto(x, y)                         // y = A * x
a.gemmInto(b, product)                   // product = A * B
a.ger(x, x, alpha = -0.5)                // A -= 0.5 * x * xT
a.syr(x)                                // Lower triangle of A += x * xT
val total = a.sum()                      // Sum of all entries
val columnSums = a.sum(axis = 0)          // Reduce rows, one result per column
val rowSums = a.sum(axis = 1)             // Reduce columns, one result per row
```

`sum()` and `sum(axis)` also accept sparse and custom matrices; sparse sums visit stored entries
without densifying. Axis sums return a fresh dense vector, with zero for each empty reduction.
`trsv` and `trsm` solve triangular systems in place with vector and matrix right-hand sides:

```kotlin
a.trsv(x, lower = true)                  // Solve A * x = b with b initially in x
a.trsm(b, lower = true)                  // Solve A * X = B with B initially in b
```

These conveniences use the default engine, accept matrix views, and preserve dense vector offsets
and strides. Vector lengths must match the matrix
dimensions exactly. Wrap a whole array with `DenseVector.wrap` to borrow it without copying.
Use `KoblasEngine` members for explicit engines and the full dense and sparse BLAS surface.

## Reusing arrays

For repeated work, write into an existing destination instead of allocating a result each time.
The names come from BLAS: GEMV means general matrix-vector multiplication, and GEMM means general
matrix-matrix multiplication.

```kotlin
val destination = DoubleArray(a.rows)
koblas.gemv(1.0, a, x.values, x.offset, x.stride, 0.0, destination, 0, 1) // y = a * x
```

Dense vector arguments specify their backing array, offset and stride. A whole array starts at zero
with stride one. These calls can scale the product and add it to the destination in one call. For matrices,
GEMM computes `c = alpha * a * b + beta * c` when neither operand is transposed. Pass a `Scratch`
to reuse temporary buffers across calls:

```kotlin
val result = DenseMatrix.zero(a.rows, b.cols)
val scratch = Scratch()

repeat(1_000) {
    a.gemmInto(b, result, scratch = scratch)
}
```

With beta set to zero, each call replaces the result; setting it to one accumulates into the result.
Transpose flags let you multiply by a transposed operand without creating a separate transposed matrix.
A `Scratch` lends temporary buffers to one call at a time and keeps them for the next. Calls returning
new results still allocate their output storage. Concurrent calls need separate destinations and
separate `Scratch` instances.

You can also reuse storage you already have. The wrapping factories share the supplied array without
copying it. Dense matrix storage is column-major: elements in a column are adjacent, and `a[i, j]`
is stored at `i + j * a.rows`. Use `ofRows` when starting with rows; use `wrap` when your flat array
already has this layout.

A view borrows a rectangular part of a matrix without copying, much like an array slice:

```kotlin
val firstColumn = a.view(columnOffset = 0, cols = 1)
firstColumn[0, 0] = 4.0 // Also changes a[0, 0]
```

Every dense BLAS routine accepts views directly, including dense operands in sparse products.
A view itself shares storage, but an operation on it can still need temporary buffers. Built-in calls
handle overlapping inputs and destinations by taking snapshots where needed. Direct vendor bindings
have their own overlap rules.

Operations follow the usual BLAS numerical contracts. Different engines can round results differently;
bit-for-bit agreement is not promised. Shapes and arguments are validated, but routines do not scan
matrices for singularity or poor conditioning.

## Java and mobile projects

From Java, factories are static methods and container conveniences use `Koblas`.
BLAS calls use `KoblasEngine` members, with `Koblas.getDefault()` selecting the default engine.
Pass every argument from Java, including transpose flags and optional scratch:

```java
import com.eignex.koblas.DenseMatrix;
import com.eignex.koblas.Koblas;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {{2.0, 1.0}, {1.0, 3.0}});
double[] x = new double[] {3.0, 5.0};
double[] destination = new double[a.getRows()];
Koblas.getDefault().gemv(1.0, a, x, 0, 1, 0.0, destination, 0, 1, false, null);
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
