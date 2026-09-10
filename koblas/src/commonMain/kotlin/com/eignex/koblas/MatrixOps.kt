@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.
@file:kotlin.jvm.JvmName("MatrixOpsKt")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.applyBeta
import com.eignex.koblas.dense.denseStoredGemvUpdate
import com.eignex.koblas.dense.denseSymmetricStoredGemvUpdate
import com.eignex.koblas.dense.genericRankOneUpdate
import com.eignex.koblas.dense.genericStoredGemvUpdate
import com.eignex.koblas.sparse.internal.sparseSyr
import com.eignex.koblas.sparse.internal.sparseSyr2

/**
 * `y = alpha * A * x + beta * y` (BLAS `dgemv`) into [destination], for any [MatrixLike] against any
 * [VectorLike]. `beta == 0.0` overwrites [destination] without reading it, so a destination left holding
 * NaN still yields a clean product.
 *
 * A sparse or generic [x] is never materialised as a dense array: a dense `A` takes one column axpy per
 * stored entry of [x], a sparse `A` walks the stored entries of each such column, and any other
 * [MatrixLike] falls back to indexed reads. Dense storage on both sides dispatches straight to the
 * backend, either [Blas.gemv] or [com.eignex.koblas.sparse.SparseBlas.gemv].
 *
 * [destination] must not be the backing array of [x] or of a dense `A`, as for [Blas.gemv] over strided
 * views: the product reads every operand entry while writing, so an aliased destination would feed partial
 * results back into the sum.
 */
public fun MatrixLike.gemvInto(alpha: Double, x: VectorLike, beta: Double, destination: DoubleArray) {
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
    if (x is DenseVector && a is DenseMatrix) {
        koblas.gemv(alpha, a, x.data, beta, destination)
        return
    }
    if (x is DenseVector && a is SparseMatrix) {
        koblas.sparseBlas.gemv(alpha, a, x.data, beta, destination)
        return
    }
    destination.prescale(beta)
    if (alpha == 0.0) return
    when (a) {
        is DenseMatrix -> {
            denseStoredGemvUpdate(koblas.denseKernelFamilies.vector, alpha, a, x, destination)
        }

        is SparseMatrix -> x.forEachStored { j, v ->
            if (v != 0.0) {
                val scaled = alpha * v
                koblas.sparseKernelFamilies.indexed.axpy(
                    a.rowIdx,
                    a.values,
                    a.colPtr[j],
                    a.colPtr[j + 1],
                    scaled,
                    destination,
                )
            }
        }

        else -> genericStoredGemvUpdate(alpha, a, x, destination)
    }
}

/** [gemvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
public fun MatrixLike.gemvInto(x: VectorLike, destination: DoubleArray): Unit = gemvInto(1.0, x, 0.0, destination)

/**
 * `y = alpha * A * x + beta * y` for a symmetric `A` (BLAS `dsymv`) into [destination], accepting any
 * [VectorLike] for [x]. Only the [lower] triangle is read, diagonal included, and `beta == 0.0`
 * overwrites [destination] without reading it.
 *
 * Reading one triangle is what lets a caller maintain its symmetric matrix with [DenseMatrix.syr], which
 * touches half the entries, rather than the full-matrix [DenseMatrix.ger]. Outside the stored triangle
 * each entry is taken from its mirror, so the other half may hold anything.
 */
@Suppress("LongParameterList") // the BLAS dsymv signature
public fun DenseMatrix.symvInto(
    alpha: Double,
    x: VectorLike,
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
    if (x is DenseVector) {
        koblas.symv(alpha, this, x.data, beta, destination, lower)
        return
    }
    destination.prescale(beta)
    if (alpha == 0.0) return
    denseSymmetricStoredGemvUpdate(alpha, this, x, destination, lower)
}

/** [symvInto] with `alpha = 1, beta = 0`, so `destination` receives `A * x`. */
public fun DenseMatrix.symvInto(x: VectorLike, destination: DoubleArray, lower: Boolean = true): Unit =
    symvInto(1.0, x, 0.0, destination, lower)

/** The `beta * y` half of a matvec. A zero [beta] overwrites without reading, as BLAS specifies, so the
 *  destination's previous contents cannot poison the result. */
private fun DoubleArray.prescale(beta: Double) = applyBeta(koblas.denseKernelFamilies.vector, this, 0, size, beta)

/** Whether [destination] is the very array this vector is stored in. */
private fun VectorLike.sharesStorage(destination: DoubleArray): Boolean = this is DenseVector && data === destination

/** Whether [destination] is the very array this matrix is stored in. */
private fun MatrixLike.sharesStorage(destination: DoubleArray): Boolean = this is DenseMatrix && data === destination

/**
 * Rank-one update `A = A + alpha * x * yT` (BLAS `dger`) in place. Subtract by passing
 * `alpha = -1.0`.
 */
public fun DenseMatrix.ger(alpha: Double, x: VectorLike, y: VectorLike) {
    requireShape(rows == x.size && cols == y.size) {
        "ger shape mismatch: A is ${rows}x$cols, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is DenseVector && y is DenseVector) {
        koblas.ger(alpha, x.data, y.data, this)
        return
    }
    genericRankOneUpdate(alpha, x, y, this)
}

/** Symmetric rank-1 update `A += alpha * x * xT` (BLAS `dsyr`) in place. See [Blas.syr]. */
public fun DenseMatrix.syr(alpha: Double, x: VectorLike, lower: Boolean = true): Unit = koblas.syr(
    alpha,
    x,
    this,
    lower,
)

/** Symmetric rank-2 update `A += alpha * (x * yT + y * xT)` (BLAS `dsyr2`) in place. See [Blas.syr2]. */
public fun DenseMatrix.syr2(alpha: Double, x: VectorLike, y: VectorLike, lower: Boolean = true): Unit =
    koblas.syr2(alpha, x, y, this, lower)

/**
 * Fresh CSC matrix holding `A + alpha * x * xT` in its [lower] or upper triangle. The other triangle is
 * copied unchanged. Unlike dense [DenseMatrix.syr], this is not in place: a rank update can introduce
 * entries that the source CSC pattern has no room to store.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * arithmetic cancels or underflows to zero, so the returned matrix never silently drops discovered fill.
 * The result owns independent structural and value arrays, and its rows ascend within every column.
 */
public fun SparseMatrix.syr(alpha: Double, x: VectorLike, lower: Boolean = true): SparseMatrix {
    requireSyrShape(this, x.size, "syr")
    return sparseSyr(this, alpha, x, lower)
}

/**
 * Fresh CSC matrix holding `A + alpha * (x * yT + y * xT)` in its [lower] or upper triangle. The other
 * triangle is copied unchanged. This structural counterpart of dense [DenseMatrix.syr2] never mutates
 * its source, because newly nonzero entries may require CSC fill.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * two terms cancel or underflow to zero. The result has independent arrays and canonical ascending CSC rows.
 */
public fun SparseMatrix.syr2(alpha: Double, x: VectorLike, y: VectorLike, lower: Boolean = true): SparseMatrix {
    requireSyr2Shape(this, x.size, y.size, "syr2")
    return sparseSyr2(this, alpha, x, y, lower)
}
