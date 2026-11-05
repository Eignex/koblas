@file:Suppress("TooManyFunctions") // Storage dispatch for the vector convenience API.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.engine.BuiltinEngine
import com.eignex.koblas.internal.kernels.scalar.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.storage.sameVectorMapping

/** Storage adapters use selected array kernels where possible and portable readers for opaque storage. */
internal fun KoblasEngine.dotVector(x: Vector, y: Vector): Double {
    requireSameSize(x.size, y.size, "dot")
    return when {
        x is DenseVector && y is DenseVector -> dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)

        x is SparseVector && y is SparseVector -> dot(
            x.indices, 0, x.values, 0, x.values.size, y.indices, 0, y.values, 0, y.values.size,
        )

        x is SparseVector && y is DenseVector -> if (y.isWholeArray) {
            dot(x.indices, 0, x.values, 0, x.values.size, y.values)
        } else {
            indexedDotDense(x.indices, 0, x.values, 0, x.values.size, y.values, y.offset, y.stride)
        }

        y is SparseVector -> dotVector(y, x)

        else -> scalarVectorDot(x, y)
    }
}

internal fun KoblasEngine.normVector(x: Vector): Double = when (x) {
    is DenseVector -> nrm2(x.values, x.offset, x.size, x.stride)

    is SparseVector -> nrm2(x.values, 0, x.values.size)

    // Opaque storage needs the portable rescaling path and an owned snapshot.
    else -> ScalarEngine.nrm2(x.toDoubleArray(), 0, x.size)
}

internal fun KoblasEngine.sumVector(x: Vector): Double = when (x) {
    is DenseVector -> sum(x.values, x.offset, x.size, x.stride)
    is SparseVector -> sum(x.values, 0, x.values.size)
    else -> scalarVectorSum(x)
}

internal fun KoblasEngine.absoluteSumVector(x: Vector): Double = when (x) {
    is DenseVector -> asum(x.values, x.offset, x.size, x.stride)
    is SparseVector -> asum(x.values, 0, x.values.size)
    else -> scalarVectorAbsoluteSum(x)
}

internal fun KoblasEngine.copyVector(x: Vector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    if (x is DenseVector) return copyDense(x, y)
    when (val source = x.stableFor(y)) {
        is SparseVector -> {
            zeroVector(y.values, y.offset, y.size, y.stride)
            if (y.isWholeArray) {
                scatter(source.indices, 0, source.values, 0, source.values.size, y.values)
            } else {
                indexedScatter(source.indices, 0, source.values, 0, source.values.size, y.values, y.offset, y.stride)
            }
        }

        is DenseVector -> copyDense(source, y)
    }
}

private fun KoblasEngine.copyDense(x: DenseVector, y: DenseVector) {
    requireSameSize(x.size, y.size, "copy")
    if (sameVectorMapping(x, y)) return
    val contiguous = x.stride == 1 && y.stride == 1
    val provider = if (this is BuiltinEngine) vectorProvider(EngineImplementation.Copy, x.size, contiguous) else this
    // Portable contiguous copy is a memmove, so it needs no alias snapshot.
    val source = if (provider === ScalarEngine && contiguous) x else x.stableFor(y)
    provider.copy(source.values, source.offset, y.values, y.offset, x.size, source.stride, y.stride)
}

internal fun KoblasEngine.axpyVector(alpha: Double, x: Vector, y: DenseVector) {
    requireSameSize(x.size, y.size, "axpy")
    if (alpha == 0.0) return
    when (val source = x.stableFor(y)) {
        is SparseVector -> if (y.isWholeArray) {
            axpy(source.indices, 0, source.values, 0, source.values.size, alpha, y.values)
        } else {
            indexedAxpy(source.indices, 0, source.values, 0, source.values.size, alpha, y.values, y.offset, y.stride)
        }

        is DenseVector -> {
            requireSameSize(source.size, y.size, "axpy")
            axpy(y.values, y.offset, alpha, source.values, source.offset, source.size, y.stride, source.stride)
        }
    }
}
