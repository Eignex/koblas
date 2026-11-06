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
koblas.gemv(1.0, a, x, 0.0, y)
val product = DenseMatrix.zero(a.rows, b.cols)
koblas.gemm(1.0, a, false, b, false, 0.0, product)
val snapshot = x.copy()
val squaredLength = x dot x
val length = x.norm()              // Euclidean norm, Norm.Two
val absoluteSum = x.norm(Norm.One)
val largestMagnitude = x.norm(Norm.Infinity)
val matrixLength = a.norm()        // Frobenius norm
val largestEntry = a.norm(Norm.MaxAbs)
```

GEMV writes a matrix-vector product into `y`; GEMM writes a matrix product into `product`.
`copy()` returns independent storage, and a dot product returns a scalar. Access individual elements
with `a[row, column]` or `x[index]`.

Dense containers store every element. Sparse containers store only selected positions and are useful
when most entries are absent. Matrix products accept a sparse operand on either side of a dense operand
and produce a dense result. Sparse inputs stay sparse during the computation. Products of two sparse
matrices are unsupported. See the [API overview](koblas/module.md) for construction and the full
operation set.

The container facades work through the mutable `Matrix` and `Vector` contracts, including sparse,
strided and custom storage:

```kotlin
val vectorCopy = x.copy()
val matrixCopy = a.copy()
vectorCopy.copyFrom(x)
matrixCopy.copyFrom(b)
val transposed = a.transpose()
val total = a.sum()
val columnSums = a.sum(axis = 0) // Reduce rows, one result per column
val rowSums = a.sum(axis = 1)    // Reduce columns, one result per row
```

`copy()` and `transpose()` return independent storage. Sparse sources keep their stored zeros;
dense views and custom sources produce owning dense results. `copyFrom()` validates equal logical
shapes before writing and handles aliases, including overlapping views. A sparse destination keeps
a sparse source's pattern; other sources store nonzero entries and negative zeros. Sparse insertion
keeps indices sorted, and structural writes may replace the arrays returned by sparse accessors.

Axis sums return a fresh dense vector. Sparse reductions visit stored entries without densifying.
BLAS arithmetic uses typed `KoblasEngine` members with explicit destinations. Dense and sparse
containers are public operands; primitive array windows belong to internal kernels.
`vector.scale(alpha)` and `destination.axpy(alpha, source)` use the default engine across vector storage.

Vectors and matrices share the `Norm` enum. Vectors support `One`, `Two` (the default), `Infinity`
and `MaxAbs`, with the last two equivalent. Matrices support `Frobenius` (the default), `One` for the
largest absolute column sum, `Infinity` for the largest absolute row sum, and `MaxAbs` for the largest
absolute entry. Matrix `Two` and vector `Frobenius` throw `IllegalArgumentException`. Norms accept
dense, sparse and custom storage, and empty inputs have norm zero for every supported choice.

## Reusing arrays

For repeated work, write into an existing destination instead of allocating a result each time.
The names come from BLAS: GEMV means general matrix-vector multiplication, and GEMM means general
matrix-matrix multiplication.

```kotlin
val destination = DenseVector.zero(a.rows)
koblas.gemv(1.0, a, x, 0.0, destination) // y = a * x
```

Dense vector arguments carry their own offsets and strides, including borrowed views. Wrap an existing
array with `DenseVector.wrap` to reuse it. These calls can scale the product and add it to the destination
in one call. For matrices,
GEMM computes `c = alpha * a * b + beta * c` when neither operand is transposed. Pass a `Scratch`
to reuse temporary buffers across calls:

```kotlin
val result = DenseMatrix.zero(a.rows, b.cols)
val scratch = Scratch()

repeat(1_000) {
    koblas.gemm(1.0, a, false, b, false, 0.0, result, scratch = scratch)
}
```

With beta set to zero, each call replaces the result; setting it to one accumulates into the result.
Transpose flags let you multiply by a transposed operand without creating a separate transposed matrix.
A `Scratch` lends temporary buffers to one call at a time and keeps them for the next. Calls returning
new results still allocate their output storage. Concurrent calls need separate destinations and
separate `Scratch` instances.

You can also reuse storage you already have. The wrapping factories share the supplied array without
copying it. Dense matrix storage is column-major: elements in a column are adjacent, and `a[i, j]`
is stored at `a.offset + i + j * a.leadingDimension`. Allocating factories use offset zero and a leading
dimension equal to the row count. Use `ofRows` for rows, or `wrap` for an existing buffer with optional
offset and column spacing.

A view borrows a rectangular part of a matrix without copying, much like an array slice:

```kotlin
val firstColumn = a.view(columnOffset = 0, cols = 1)
firstColumn[0, 0] = 4.0 // Also changes a[0, 0]
```

`DenseMatrix` represents both whole buffers and views; `view()` returns the same type. `DenseVector`
likewise represents whole buffers and strided slices, including reversed slices:

```kotlin
val reversed = x.view(offset = x.size - 1, size = x.size, stride = -1)
```

Use `DenseVector.wrap(buffer, offset, size, stride)` to borrow an array window. Dense equality and
hashing compare logical entries, and serialization writes only those entries, decoding into compact,
independent storage. `MatrixStorage` and `VectorStorage` support polymorphic dense/sparse serialization.

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
import com.eignex.koblas.DenseVector;
import com.eignex.koblas.Koblas;

DenseMatrix a = DenseMatrix.ofRows(new double[][] {{2.0, 1.0}, {1.0, 3.0}});
DenseVector x = DenseVector.of(new double[] {3.0, 5.0});
DenseVector destination = DenseVector.zero(a.getRows());
Koblas.getDefault().gemv(1.0, a, x, 0.0, destination, false, null);
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
