@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.operations.copyIntoMatrix
import com.eignex.koblas.internal.operations.normMatrix
import com.eignex.koblas.internal.operations.sumMatrix
import com.eignex.koblas.internal.storage.copyMatrixStorage
import com.eignex.koblas.internal.storage.transposeMatrixStorage

/**
 * A writable vector view of [column]. Mutations update this matrix; the vector's indices are local rows.
 * Built-in containers return [DenseVector] or [SparseVector]; other implementations use [Matrix.get]
 * and [Matrix.set]. Use [Vector.copy] for independent storage.
 */
public fun Matrix.columnView(column: Int): Vector {
    if (column !in 0 until cols) throw IndexOutOfBoundsException("column $column outside [0,$cols)")
    return when (this) {
        is DenseMatrix -> columnView(column)
        is SparseMatrix -> columnView(column)
        else -> com.eignex.koblas.internal.storage.MatrixColumnVector(this, column)
    }
}

/**
 * Visits writable column views in column order, including empty columns. Mutations through a callback's
 * vector update this matrix. Concrete containers retain their dense or sparse callback type for calls to
 * [KoblasEngine]; through [Matrix], callbacks use the generic [Vector] contract. Returns no new matrix.
 */
@kotlin.jvm.JvmSynthetic
public inline fun Matrix.forEachColumns(block: (column: Vector) -> Unit) {
    for (j in 0 until cols) block(columnView(j))
}

/**
 * Visits stored entries as `(row, column, value)`, columns first and rows ascending within each column.
 * [SparseMatrix] visits only stored positions, including stored zeros; other implementations visit every
 * entry. Dense views report local indices and exclude padding. Empty matrices invoke no callbacks.
 */
@kotlin.jvm.JvmSynthetic
public inline fun Matrix.forEachStored(block: (row: Int, col: Int, value: Double) -> Unit) {
    if (rows == 0 || cols == 0) return
    when (this) {
        is DenseMatrix -> {
            val data = values
            for (j in 0 until cols) {
                var p = offset + j * leadingDimension
                for (i in 0 until rows) block(i, j, data[p++])
            }
        }

        is SparseMatrix -> for (j in 0 until cols) forEachInColumn(j) { i, value -> block(i, j, value) }

        else -> for (j in 0 until cols) for (i in 0 until rows) block(i, j, this[i, j])
    }
}

/**
 * Fresh transposed matrix. Sparse storage remains CSC and preserves stored zeros; other implementations
 * produce dense storage. Views transpose only their logical entries. For products, prefer the transpose
 * flags on [KoblasEngine.gemv] and [KoblasEngine.gemm] to avoid materializing the transpose.
 */
public fun Matrix.transpose(): Matrix = transposeMatrixStorage(this)

/**
 * An independent copy of the logical entries. Sparse storage keeps its pattern, including stored zeros;
 * dense views and custom matrices produce owning dense storage. No backing arrays are shared.
 */
public fun Matrix.copy(): Matrix = copyMatrixStorage(this)

/**
 * Copies [source] into this matrix, requiring equal shapes before writing. Inputs may share storage;
 * overlapping or opaque sources are snapshotted first. Entries outside a destination view are untouched.
 *
 * Sparse destinations retain a sparse source's pattern and stored zeros. Other sources store nonzero
 * entries and negative zeros. Structural changes replace sparse buffers; equal patterns reuse them.
 */
public fun Matrix.copyFrom(source: Matrix): Unit = copyIntoMatrix(source, this)

/**
 * Matrix norm of [type]: [Norm.Frobenius] is the rescaled square root of the sum of squared entries,
 * [Norm.One] is the largest absolute column sum, [Norm.Infinity] is the largest absolute row sum, and
 * [Norm.MaxAbs] is the maximum absolute entry. A matrix with either dimension empty has norm zero.
 *
 * Uses the default [koblas] engine for dense and packed sparse reductions and portable readers for
 * custom matrices. Views exclude padding and sparse operands are never densified. Rounding and
 * non-finite behavior follow the selected implementation.
 *
 * @throws IllegalArgumentException if [type] is [Norm.Two]; the spectral matrix norm is not supported.
 */
public fun Matrix.norm(type: Norm = Norm.Frobenius): Double = koblas.normMatrix(this, type)

/**
 * Plain sum of all entries. Dense storage uses [KoblasEngine.sum] through the default [koblas] engine,
 * sparse storage sums only its stored values, and custom matrices are read through [Matrix.get].
 * An empty matrix sums to zero. Summation order and rounding depend on the storage and engine.
 */
public fun Matrix.sum(): Double = koblas.sumMatrix(this)

/**
 * Sums along [axis] into a fresh dense vector: `axis = 0` reduces rows and returns [Matrix.cols]
 * column sums; `axis = 1` reduces columns and returns [Matrix.rows] row sums. A zero-length reduced
 * dimension contributes zero to each result. The result owns its storage.
 *
 * Dense views address only their logical entries. Sparse matrices visit only stored entries without
 * densifying; custom matrices are read through [Matrix.get]. Summation order and rounding depend on
 * the storage and default [koblas] engine.
 *
 * @throws IllegalArgumentException if [axis] is neither zero nor one.
 */
public fun Matrix.sum(axis: Int): DenseVector = DenseVector.wrap(koblas.sumMatrix(this, axis))
