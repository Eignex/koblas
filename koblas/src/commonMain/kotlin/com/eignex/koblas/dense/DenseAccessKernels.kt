@file:Suppress("LongParameterList")

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.MatrixLike
import com.eignex.koblas.StridedMatrixView
import com.eignex.koblas.StridedVectorView
import com.eignex.koblas.VectorLike
import com.eignex.koblas.forEachStored

/** Scales a strided destination, overwriting it without reads when [beta] is zero. */
internal fun scaleStrided(y: StridedVectorView, beta: Double) {
    val yd = y.data
    var yi = y.offset
    repeat(y.size) {
        yd[yi] = when (beta) {
            0.0 -> 0.0
            1.0 -> yd[yi]
            else -> beta * yd[yi]
        }
        yi += y.stride
    }
}

/**
 * Adds `alpha * op(A) * x` to a strided destination after validation and beta scaling by the caller.
 * All logical dimensions must be positive. The non-transposed column coefficient is evaluated even when
 * it is zero, retaining dense `0 * infinity` arithmetic without materialising any view.
 */
internal fun stridedGemvUpdate(
    alpha: Double,
    a: StridedMatrixView,
    x: StridedVectorView,
    y: StridedVectorView,
    transpose: Boolean,
) {
    val ad = a.data
    val xd = x.data
    val yd = y.data
    if (transpose) {
        var aColumn = a.offset
        var yi = y.offset
        repeat(a.cols) {
            var sum = 0.0
            var ai = aColumn
            var xi = x.offset
            repeat(a.rows) {
                sum += ad[ai] * xd[xi]
                ai++
                xi += x.stride
            }
            yd[yi] += alpha * sum
            aColumn += a.leadingDimension
            yi += y.stride
        }
    } else {
        var aColumn = a.offset
        var xi = x.offset
        repeat(a.cols) {
            val multiplier = alpha * xd[xi]
            var ai = aColumn
            var yi = y.offset
            repeat(a.rows) {
                yd[yi] += multiplier * ad[ai]
                ai++
                yi += y.stride
            }
            aColumn += a.leadingDimension
            xi += x.stride
        }
    }
}

/** Scales a strided matrix destination, overwriting entries without reads when [beta] is zero. */
internal fun scaleStrided(c: StridedMatrixView, beta: Double) {
    val cd = c.data
    var column = c.offset
    repeat(c.cols) {
        var ci = column
        repeat(c.rows) {
            cd[ci] = when (beta) {
                0.0 -> 0.0
                1.0 -> cd[ci]
                else -> beta * cd[ci]
            }
            ci++
        }
        column += c.leadingDimension
    }
}

/**
 * Writes `alpha * op(A) * op(B) + beta * C` to strided column-major panels without copying them.
 * The caller owns validation and excludes a zero alpha or shared dimension, for which only scaling is needed.
 */
internal fun stridedGemmUpdate(
    alpha: Double,
    a: StridedMatrixView,
    transposeA: Boolean,
    b: StridedMatrixView,
    transposeB: Boolean,
    beta: Double,
    c: StridedMatrixView,
    m: Int,
    k: Int,
    n: Int,
) {
    val ad = a.data
    val bd = b.data
    val cd = c.data
    repeat(n) { j ->
        repeat(m) { i ->
            var sum = 0.0
            repeat(k) { p ->
                val ai = a.offset + if (transposeA) p + i * a.leadingDimension else i + p * a.leadingDimension
                val bi = b.offset + if (transposeB) j + p * b.leadingDimension else p + j * b.leadingDimension
                sum += ad[ai] * bd[bi]
            }
            val ci = c.offset + i + j * c.leadingDimension
            cd[ci] = alpha * sum + when (beta) {
                0.0 -> 0.0
                1.0 -> cd[ci]
                else -> beta * cd[ci]
            }
        }
    }
}

/** Adds dense matrix columns selected by the stored entries of [x]. */
internal fun denseStoredGemvUpdate(
    kernels: DenseVectorKernels,
    alpha: Double,
    a: DenseMatrix,
    x: VectorLike,
    destination: DoubleArray,
) {
    val ad = a.data
    val rows = a.rows
    x.forEachStored { j, value ->
        if (value != 0.0) kernels.axpy(destination, 0, alpha * value, ad, j * rows, rows)
    }
}

/** Adds a generic indexed matrix-vector product, visiting only entries represented by [x]. */
internal fun genericStoredGemvUpdate(alpha: Double, a: MatrixLike, x: VectorLike, destination: DoubleArray) {
    repeat(a.rows) { i ->
        var sum = 0.0
        x.forEachStored { j, value -> sum += a[i, j] * value }
        destination[i] += alpha * sum
    }
}

/** Adds a selected-triangle symmetric dense product for a non-dense vector operand. */
internal fun denseSymmetricStoredGemvUpdate(
    alpha: Double,
    a: DenseMatrix,
    x: VectorLike,
    destination: DoubleArray,
    lower: Boolean,
) {
    val ad = a.data
    val n = a.rows
    x.forEachStored { j, value ->
        if (value != 0.0) {
            val scaled = alpha * value
            repeat(n) { i ->
                val index = if (lower == (i >= j)) i + j * n else j + i * n
                destination[i] += ad[index] * scaled
            }
        }
    }
}

/** Applies a generic stored-entry rank-one update to a dense column-major destination. */
internal fun genericRankOneUpdate(alpha: Double, x: VectorLike, y: VectorLike, a: DenseMatrix) {
    val ad = a.data
    val rows = a.rows
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val column = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> ad[column + i] += scaled * xi }
        }
    }
}

/** Returns an existing contiguous dense buffer or stages represented entries into a fresh owned buffer. */
internal fun contiguousVectorData(x: VectorLike): DoubleArray = when (x) {
    is DenseVector -> x.data

    else -> DoubleArray(x.size).also { destination ->
        x.forEachStored { index, value -> destination[index] = value }
    }
}
