@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

// Part of the MatrixOpsKt facade. Splitting the file would otherwise rename the class JVM callers
// compiled against, so the four parts are joined back into one rather than becoming four.

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.sparse.internal.sparseSyr
import com.eignex.koblas.sparse.internal.sparseSyr2

/**
 * `y = alpha * A * x + beta * y` (BLAS `dgemv`) into [destination], for any [F64MatrixLike] against any
 * [F64VectorLike]. `beta == 0.0` overwrites [destination] without reading it, so a destination left holding
 * NaN still yields a clean product.
 *
 * A sparse or generic [x] is never materialised as a dense array: a dense `A` takes one column axpy per
 * stored entry of [x], a sparse `A` walks the stored entries of each such column, and any other
 * [F64MatrixLike] falls back to indexed reads. Dense storage on both sides dispatches straight to the
 * backend, either [F64Blas.gemv] or [com.eignex.koblas.sparse.F64SparseBlas.gemv].
 *
 * [destination] must not be the backing array of [x] or of a dense `A`, as for [F64Blas.gemv] over strided
 * views: the product reads every operand entry while writing, so an aliased destination would feed partial
 * results back into the sum.
 */
public fun F64MatrixLike.gemvInto(alpha: Double, x: F64VectorLike, beta: Double, destination: DoubleArray) {
    val a = this
    requireShape(a.cols == x.size) { "gemvInto shape mismatch: A is ${a.rows}x${a.cols}, x size ${x.size}" }
    requireShape(destination.size == a.rows) {
        "gemvInto: destination size ${destination.size} != rows ${a.rows}"
    }
    require(!x.sharesStorage(destination) && !a.sharesStorage(destination)) {
        "gemvInto: destination overlaps an input"
    }
    // The seams quick-return on a zero-extent operand before scaling, which is netlib's rule for gemv but
    // not the contract above: this one promises that `beta == 0.0` overwrites a destination that may arrive
    // holding NaN. Settling it here keeps every storage combination answering the same way, where otherwise
    // a dense 3x0 left the NaN in place and a generic one returned zeros.
    if (a.cols == 0) {
        destination.prescale(beta)
        return
    }
    if (x is F64DenseVector && a is F64DenseMatrix) {
        koblas.gemv(alpha, a, x.data, beta, destination)
        return
    }
    if (x is F64DenseVector && a is F64SparseMatrix) {
        koblas.sparseBlas.gemv(alpha, a, x.data, beta, destination)
        return
    }
    destination.prescale(beta)
    if (alpha == 0.0) return
    when (a) {
        is F64DenseMatrix -> {
            val ad = a.data
            val rows = a.rows
            // Read the installed kernels once rather than per stored entry of x.
            val kernels = koblas.kernels
            x.forEachStored { j, v ->
                if (v != 0.0) kernels.axpy(destination, 0, alpha * v, ad, j * rows, rows)
            }
        }

        is F64SparseMatrix -> x.forEachStored { j, v ->
            if (v != 0.0) {
                val scaled = alpha * v
                a.forEachInColumn(j) { i, aij -> destination[i] += aij * scaled }
            }
        }

        else -> for (i in 0 until a.rows) {
            var sum = 0.0
            x.forEachStored { j, v -> sum += a[i, j] * v }
            destination[i] += alpha * sum
        }
    }
}

/** [gemvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
public fun F64MatrixLike.gemvInto(x: F64VectorLike, destination: DoubleArray): Unit = gemvInto(1.0, x, 0.0, destination)

/**
 * `y = alpha * A * x + beta * y` for a symmetric `A` (BLAS `dsymv`) into [destination], accepting any
 * [F64VectorLike] for [x]. Only the [lower] triangle is read, diagonal included, and `beta == 0.0`
 * overwrites [destination] without reading it.
 *
 * Reading one triangle is what lets a caller maintain its symmetric matrix with [F64DenseMatrix.syr], which
 * touches half the entries, rather than the full-matrix [F64DenseMatrix.ger]. Outside the stored triangle
 * each entry is taken from its mirror, so the other half may hold anything.
 */
@Suppress("LongParameterList") // the BLAS dsymv signature
public fun F64DenseMatrix.symvInto(
    alpha: Double,
    x: F64VectorLike,
    beta: Double,
    destination: DoubleArray,
    lower: Boolean = true,
) {
    requireSquare(this, "symvInto")
    requireShape(cols == x.size) { "symvInto shape mismatch: A is ${rows}x$cols, x size ${x.size}" }
    requireShape(destination.size == rows) { "symvInto: destination size ${destination.size} != rows $rows" }
    require(!x.sharesStorage(destination) && !sharesStorage(destination)) {
        "symvInto: destination overlaps an input"
    }
    if (x is F64DenseVector) {
        koblas.symv(alpha, this, x.data, beta, destination, lower)
        return
    }
    destination.prescale(beta)
    if (alpha == 0.0) return
    x.forEachStored { j, v ->
        if (v != 0.0) {
            val scaled = alpha * v
            for (i in 0 until rows) {
                val aij = if (lower == (i >= j)) this[i, j] else this[j, i]
                destination[i] += aij * scaled
            }
        }
    }
}

/** [symvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
public fun F64DenseMatrix.symvInto(x: F64VectorLike, destination: DoubleArray, lower: Boolean = true): Unit =
    symvInto(1.0, x, 0.0, destination, lower)

/** The `beta * y` half of a matvec. A zero [beta] overwrites without reading, as BLAS specifies, so the
 *  destination's previous contents cannot poison the result. */
private fun DoubleArray.prescale(beta: Double) = applyBeta(koblas.kernels, this, 0, size, beta)

/** Whether [destination] is the very array this vector is stored in. */
private fun F64VectorLike.sharesStorage(destination: DoubleArray): Boolean =
    this is F64DenseVector && data === destination

/** Whether [destination] is the very array this matrix is stored in. */
private fun F64MatrixLike.sharesStorage(destination: DoubleArray): Boolean =
    this is F64DenseMatrix && data === destination

/**
 * Rank-one update `A = A + alpha * x * yT` (BLAS `dger`) in place. Subtract by passing
 * `alpha = -1.0`.
 */
public fun F64DenseMatrix.ger(alpha: Double, x: F64VectorLike, y: F64VectorLike) {
    requireShape(rows == x.size && cols == y.size) {
        "ger shape mismatch: A is ${rows}x$cols, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is F64DenseVector && y is F64DenseVector) {
        koblas.ger(alpha, x.data, y.data, this)
        return
    }
    val md = data
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val col = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> md[col + i] += scaled * xi }
        }
    }
}

/** Symmetric rank-1 update `A += alpha * x * xT` (BLAS `dsyr`) in place. See [F64Blas.syr]. */
public fun F64DenseMatrix.syr(alpha: Double, x: F64VectorLike, lower: Boolean = true): Unit = koblas.syr(
    alpha,
    x,
    this,
    lower,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`) in place. See [F64Blas.syr2]. */
public fun F64DenseMatrix.syr2(alpha: Double, x: F64VectorLike, y: F64VectorLike, lower: Boolean = true): Unit =
    koblas.syr2(alpha, x, y, this, lower)

/**
 * Fresh CSC matrix holding `A + alpha * x * xT` in its [lower] or upper triangle. The other triangle is
 * copied unchanged. Unlike dense [F64DenseMatrix.syr], this is not in place: a rank update can introduce
 * entries that the source CSC pattern has no room to store.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * arithmetic cancels or underflows to zero, so the returned matrix never silently drops discovered fill.
 * The result owns independent structural and value arrays, and its rows ascend within every column.
 */
public fun F64SparseMatrix.syr(alpha: Double, x: F64VectorLike, lower: Boolean = true): F64SparseMatrix {
    requireShape(rows == cols) { "syr: matrix must be square, got ${rows}x$cols" }
    requireShape(x.size == rows) { "syr: x length ${x.size} != $rows" }
    return sparseSyr(this, alpha, x, lower)
}

/**
 * Fresh CSC matrix holding `A + alpha * (x * yT + y * xT)` in its [lower] or upper triangle. The other
 * triangle is copied unchanged. This structural counterpart of dense [F64DenseMatrix.syr2] never mutates
 * its source, because newly nonzero entries may require CSC fill.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * two terms cancel or underflow to zero. The result has independent arrays and canonical ascending CSC rows.
 */
public fun F64SparseMatrix.syr2(
    alpha: Double,
    x: F64VectorLike,
    y: F64VectorLike,
    lower: Boolean = true,
): F64SparseMatrix {
    requireShape(rows == cols) { "syr2: matrix must be square, got ${rows}x$cols" }
    requireShape(x.size == rows && y.size == rows) {
        "syr2: operand lengths ${x.size} and ${y.size} must both be $rows"
    }
    return sparseSyr2(this, alpha, x, y, lower)
}
