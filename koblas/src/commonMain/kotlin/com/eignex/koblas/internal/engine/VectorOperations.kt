@file:Suppress("TooManyFunctions") // Storage dispatch for the vector convenience API.

package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.*
import com.eignex.koblas.internal.portable.sparse.*
import com.eignex.koblas.sparse.*

/** Storage adapters use selected array kernels where possible and portable readers for opaque storage. */
internal fun KoblasEngine.dotVector(x: Vector, y: Vector): Double {
    requireSameSize(x.size, y.size, "dot")
    return when {
        x is DenseVector && y is DenseVector -> dot(x, y)

        x is SparseVector && y is SparseVector -> dot(x, y)

        x is SparseVector && y is DenseVector -> if (y.isWholeArray) {
            dot(x, y.values)
        } else {
            indexedDotDense(x.indices, 0, x.values, 0, x.values.size, y.values, y.offset, y.stride)
        }

        y is SparseVector -> dotVector(y, x)

        else -> portableDot(x, y)
    }
}

internal fun KoblasEngine.normVector(x: Vector): Double = when (x) {
    is DenseVector -> nrm2(x)

    is SparseVector -> nrm2(x)

    // Keep the opaque-storage snapshot and portable rescaling used by the convenience API.
    else -> PortableEngine.nrm2(x.toDoubleArray(), 0, x.size)
}

internal fun KoblasEngine.sumVector(x: Vector): Double = when (x) {
    is DenseVector -> sum(x.values, x.offset, x.size, x.stride)
    is SparseVector -> sum(x.values, 0, x.values.size)
    else -> portableSum(x)
}

internal fun KoblasEngine.absoluteSumVector(x: Vector): Double = when (x) {
    is DenseVector -> asum(x)
    is SparseVector -> asum(x)
    else -> portableAbsoluteSum(x)
}

internal fun KoblasEngine.maxAbsVector(x: Vector): Int = when (x) {
    is DenseVector -> iamax(x)
    else -> portableMaxAbs(x)
}

internal fun KoblasEngine.copyVector(x: Vector, y: DenseVector) {
    if (x is DenseVector) return copy(x, y)
    requireSameSize(x.size, y.size, "copy")
    when (val source = x.stableFor(y)) {
        is SparseVector -> {
            zeroVector(y.values, y.offset, y.size, y.stride)
            if (y.isWholeArray) {
                scatter(source, y.values)
            } else {
                indexedScatter(source.indices, 0, source.values, 0, source.values.size, y.values, y.offset, y.stride)
            }
        }

        is DenseVector -> copy(source, y)
    }
}

internal fun KoblasEngine.axpyVector(alpha: Double, x: Vector, y: DenseVector) {
    if (x is DenseVector) return axpy(alpha, x, y)
    requireSameSize(x.size, y.size, "axpy")
    if (alpha == 0.0) return
    when (val source = x.stableFor(y)) {
        is SparseVector -> if (y.isWholeArray) {
            axpy(y.values, alpha, source)
        } else {
            indexedAxpy(source.indices, 0, source.values, 0, source.values.size, alpha, y.values, y.offset, y.stride)
        }

        is DenseVector -> axpy(alpha, source, y)
    }
}
