package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.storage.*

/** [block] over [vector], or over a contiguous snapshot of its logical entries when [aliased]. */
internal inline fun <T> stagedWindow(
    arrays: Scratch?,
    vector: DenseVector,
    aliased: Boolean,
    block: (DenseVector) -> T,
): T {
    if (!aliased) return block(vector)
    return arrays.borrow(vector.size) { copy ->
        for (i in 0 until vector.size) copy[i] = vector[i]
        block(DenseVector.wrap(copy))
    }
}

/** Snapshots [a] only when its logical entries overlap [vector]. */
internal inline fun <T> stagedRegionAgainst(
    arrays: Scratch?,
    a: DenseMatrix,
    vector: DenseVector,
    lower: Boolean? = null,
    unitDiag: Boolean = false,
    block: (DenseMatrix) -> T,
): T {
    if (!regionWindowOverlap(a, vector.values, vector.offset, vector.size, vector.stride)) return block(a)
    return stagedRegion(arrays, a, vector.values, lower = lower, unitDiag = unitDiag, block = block)
}

/** Stages both vectors against [destination], sharing a snapshot when they address the same entries. */
internal inline fun <T> stagedWindows(
    arrays: Scratch?,
    x: DenseVector,
    y: DenseVector,
    destination: DenseMatrix,
    block: (DenseVector, DenseVector) -> T,
): T {
    val xAliased = regionWindowOverlap(destination, x.values, x.offset, x.size, x.stride)
    return stagedWindow(arrays, x, xAliased) { sx ->
        if (sameVectorMapping(x, y)) {
            block(sx, sx)
        } else {
            val yAliased = regionWindowOverlap(destination, y.values, y.offset, y.size, y.stride)
            stagedWindow(arrays, y, yAliased) { sy -> block(sx, sy) }
        }
    }
}
