@file:Suppress("TooManyFunctions") // Independent windows and numerical leaves.

package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.*
import com.eignex.koblas.internal.kernels.scalar.sparse.*
import com.eignex.koblas.internal.storage.copyVectorStorage
import com.eignex.koblas.internal.storage.dense.*
import kotlin.math.abs

/** Concrete operands reach public typed operations; opaque storage uses portable readers. */
internal fun KoblasEngine.dotVector(x: Vector, y: Vector): Double = when {
    x is DenseVector && y is DenseVector -> dot(x, y)
    x is DenseVector && y is SparseVector -> dot(x, y)
    x is SparseVector && y is DenseVector -> dot(x, y)
    x is SparseVector && y is SparseVector -> dot(x, y)
    else -> dotVectorStorage(x, y)
}

/** Primitive storage leaves for the built-in typed operations, without reentering public dispatch. */
internal fun KoblasEngine.dotVectorStorage(x: Vector, y: Vector): Double {
    requireSameSize(x.size, y.size, "dot")
    return when {
        x is DenseVector && y is DenseVector -> kernels.dot(
            x.values,
            x.offset,
            y.values,
            y.offset,
            x.size,
            x.stride,
            y.stride,
        )

        x is SparseVector && y is SparseVector -> kernels.dot(
            x.indices, x.offset, x.values, x.offset, x.nnz, y.indices, y.offset, y.values, y.offset, y.nnz,
        )

        x is SparseVector && y is DenseVector -> if (y.isWholeBuffer) {
            kernels.dot(x.indices, x.offset, x.values, x.offset, x.nnz, y.values)
        } else {
            indexedDotDense(x.indices, x.offset, x.values, x.offset, x.nnz, y.values, y.offset, y.stride)
        }

        y is SparseVector -> dotVectorStorage(y, x)

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
    is DenseVector -> nrm2(x)

    is SparseVector -> nrm2(x)

    // Opaque storage needs the portable rescaling path and an owned snapshot.
    else -> ScalarEngine.nrm2(x.toDoubleArray(), 0, x.size)
}

private fun KoblasEngine.maximumMagnitudeVector(x: Vector): Double = when (x) {
    is DenseVector -> maximumMagnitude(x)
    is SparseVector -> maximumMagnitude(x.values, x.offset, x.nnz, 1)
    else -> scalarVectorMaximumMagnitude(x)
}

internal fun KoblasEngine.maximumMagnitude(values: DoubleArray, offset: Int, size: Int, stride: Int): Double =
    maximumMagnitude(DenseVector.wrap(values, offset, size, stride))

private fun KoblasEngine.maximumMagnitude(vector: DenseVector): Double {
    if (vector.size == 0) return 0.0
    return abs(vector[iamax(vector)])
}

internal fun KoblasEngine.sumVector(x: Vector): Double = when (x) {
    is DenseVector -> sum(x)
    is SparseVector -> sum(x)
    else -> scalarVectorSum(x)
}

internal fun KoblasEngine.absoluteSumVector(x: Vector): Double = when (x) {
    is DenseVector -> asum(x)
    is SparseVector -> asum(x)
    else -> scalarVectorAbsoluteSum(x)
}

internal fun KoblasEngine.scaleVector(alpha: Double, x: Vector) {
    if (alpha == 1.0) return
    when (x) {
        is DenseVector -> kernels.scale(x.values, x.offset, alpha, x.size, x.stride)
        is SparseVector -> kernels.scale(x.values, x.offset, alpha, x.nnz)
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
        is SparseVector -> if (y.isWholeBuffer) {
            kernels.axpy(source.indices, source.offset, source.values, source.offset, source.nnz, alpha, y.values)
        } else {
            indexedAxpy(
                source.indices, source.offset, source.values, source.offset, source.nnz,
                alpha, y.values, y.offset, y.stride,
            )
        }

        is DenseVector -> {
            requireSameSize(source.size, y.size, "axpy")
            kernels.axpy(y.values, y.offset, alpha, source.values, source.offset, source.size, y.stride, source.stride)
        }
    }
}

internal fun KoblasEngine.scatterVector(source: SparseVector, destination: DenseVector) {
    requireSameSize(source.size, destination.size, "scatter")
    val aliased = source.values === destination.values
    val values = if (aliased) source.values.copyOfRange(source.offset, source.offset + source.nnz) else source.values
    scatterStored(source, destination, values, if (aliased) 0 else source.offset)
}

/** Writes coefficients already staged against [destination], preserving absent positions. */
internal fun KoblasEngine.scatterStored(
    source: SparseVector,
    destination: DenseVector,
    values: DoubleArray = source.values,
    valueOffset: Int = source.offset,
) {
    if (destination.isWholeBuffer) {
        kernels.scatter(source.indices, source.offset, values, valueOffset, source.nnz, destination.values)
    } else {
        indexedScatter(
            source.indices,
            source.offset,
            values,
            valueOffset,
            source.nnz,
            destination.values,
            destination.offset,
            destination.stride,
        )
    }
}

internal fun KoblasEngine.gatherVector(source: DenseVector, destination: SparseVector, zero: Boolean) {
    requireSameSize(source.size, destination.size, "gather")
    require(source.values !== destination.values) { "gather requires distinct coefficient buffers" }
    if (source.isWholeBuffer) {
        if (zero) {
            kernels.gatherZero(
                destination.indices,
                destination.offset,
                destination.values,
                destination.offset,
                destination.nnz,
                source.values,
            )
        } else {
            kernels.gather(
                destination.indices,
                destination.offset,
                destination.values,
                destination.offset,
                destination.nnz,
                source.values,
            )
        }
    } else {
        for (k in destination.offset until destination.offset + destination.nnz) {
            val i = destination.indices[k]
            destination.values[k] = source[i]
            if (zero) source[i] = 0.0
        }
    }
}
