package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.sparse.indexedScatter
import com.eignex.koblas.internal.storage.isWholeArray

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
    if (destination.isWholeArray) {
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
    if (source.isWholeArray) {
        if (zero) {
            kernels.gatherZero(
                destination.indices, destination.offset, destination.values, destination.offset,
                destination.nnz, source.values,
            )
        } else {
            kernels.gather(
                destination.indices, destination.offset, destination.values, destination.offset,
                destination.nnz, source.values,
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
