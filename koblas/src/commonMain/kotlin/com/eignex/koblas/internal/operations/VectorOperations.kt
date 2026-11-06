@file:Suppress("TooManyFunctions") // Storage dispatch for the vector convenience API.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.internal.engine.vectorProviderFor
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.storage.copyVectorStorage
import com.eignex.koblas.internal.storage.isWholeArray
import com.eignex.koblas.internal.storage.sameVectorMapping
import kotlin.math.abs

/** Storage adapters use selected array kernels where possible and portable readers for opaque storage. */
internal fun KoblasEngine.dotVector(x: Vector, y: Vector): Double {
    requireSameSize(x.size, y.size, "dot")
    return when {
        x is DenseVector && y is DenseVector -> dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)

        x is SparseVector && y is SparseVector -> dot(
            x.indices, x.offset, x.values, x.offset, x.nnz, y.indices, y.offset, y.values, y.offset, y.nnz,
        )

        x is SparseVector && y is DenseVector -> if (y.isWholeArray) {
            dot(x.indices, x.offset, x.values, x.offset, x.nnz, y.values)
        } else {
            indexedDotDense(x.indices, x.offset, x.values, x.offset, x.nnz, y.values, y.offset, y.stride)
        }

        y is SparseVector -> dotVector(y, x)

        else -> scalarVectorDot(x, y)
    }
}

internal fun KoblasEngine.normVector(x: Vector, type: Norm = Norm.Two): Double = when (type) {
    Norm.One -> absoluteSumVector(x)
    Norm.Two -> euclideanNormVector(x)
    Norm.Infinity, Norm.MaxAbs -> maximumMagnitudeVector(x)
    Norm.Frobenius -> throw IllegalArgumentException("Frobenius norm is only supported for matrices")
}

private fun KoblasEngine.euclideanNormVector(x: Vector): Double = when (x) {
    is DenseVector -> nrm2(x.values, x.offset, x.size, x.stride)

    is SparseVector -> nrm2(x.values, x.offset, x.nnz)

    // Opaque storage needs the portable rescaling path and an owned snapshot.
    else -> ScalarEngine.nrm2(x.toDoubleArray(), 0, x.size)
}

private fun KoblasEngine.maximumMagnitudeVector(x: Vector): Double = when (x) {
    is DenseVector -> maximumMagnitude(x.values, x.offset, x.size, x.stride)
    is SparseVector -> maximumMagnitude(x.values, x.offset, x.nnz, 1)
    else -> scalarVectorMaximumMagnitude(x)
}

internal fun KoblasEngine.maximumMagnitude(values: DoubleArray, offset: Int, size: Int, stride: Int): Double {
    if (size == 0) return 0.0
    val index = iamax(values, offset, size, stride)
    return abs(values[offset + index * stride])
}

internal fun KoblasEngine.sumVector(x: Vector): Double = when (x) {
    is DenseVector -> sum(x)
    is SparseVector -> sum(x)
    else -> scalarVectorSum(x)
}

internal fun KoblasEngine.absoluteSumVector(x: Vector): Double = when (x) {
    is DenseVector -> asum(x.values, x.offset, x.size, x.stride)
    is SparseVector -> asum(x.values, x.offset, x.nnz)
    else -> scalarVectorAbsoluteSum(x)
}

internal fun KoblasEngine.copyVector(x: Vector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    if (x is DenseVector) return copyDense(x, y)
    when (val source = x.stableFor(y)) {
        is SparseVector -> {
            zeroVector(y.values, y.offset, y.size, y.stride)
            if (y.isWholeArray) {
                scatter(source.indices, source.offset, source.values, source.offset, source.nnz, y.values)
            } else {
                indexedScatter(
                    source.indices, source.offset, source.values, source.offset, source.nnz, y.values, y.offset, y.stride,
                )
            }
        }

        is DenseVector -> copyDense(source, y)
    }
}

private fun KoblasEngine.copyDense(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    if (sameVectorMapping(x, y)) return
    val contiguous = x.stride == 1 && y.stride == 1
    val provider = vectorProviderFor(EngineImplementation.Copy, x.size, contiguous)
    // Portable contiguous copy is a memmove, so it needs no alias snapshot.
    val source = if (provider === ScalarEngine && contiguous) x else x.stableFor(y)
    provider.copy(source.values, source.offset, y.values, y.offset, x.size, source.stride, y.stride)
}

internal fun KoblasEngine.scaleVector(alpha: Double, x: Vector) {
    if (alpha == 1.0) return
    when (x) {
        is DenseVector -> scale(x.values, x.offset, alpha, x.size, x.stride)
        is SparseVector -> scale(x.values, x.offset, alpha, x.nnz)
        else -> scalarVectorScale(alpha, x)
    }
}

internal fun KoblasEngine.axpyVector(alpha: Double, x: Vector, y: Vector) {
    requireSameSize(x.size, y.size, "axpy")
    if (alpha == 0.0 || x.size == 0) return
    when (y) {
        is DenseVector -> axpyDenseVector(alpha, x, y)

        is SparseVector -> {
            // Build the result before writing: even opaque sources may borrow the destination's values.
            val result = scalarSparseVectorAxpy(alpha, x, y)
            y.replaceStorage(result.indices, result.values)
        }

        else -> scalarVectorAxpy(alpha, copyVectorStorage(x), y)
    }
}

private fun KoblasEngine.axpyDenseVector(alpha: Double, x: Vector, y: DenseVector) {
    when (val source = x.stableFor(y)) {
        is SparseVector -> if (y.isWholeArray) {
            axpy(source.indices, source.offset, source.values, source.offset, source.nnz, alpha, y.values)
        } else {
            indexedAxpy(
                source.indices, source.offset, source.values, source.offset, source.nnz, alpha, y.values, y.offset, y.stride,
            )
        }

        is DenseVector -> {
            requireSameSize(source.size, y.size, "axpy")
            axpy(y.values, y.offset, alpha, source.values, source.offset, source.size, y.stride, source.stride)
        }
    }
}
