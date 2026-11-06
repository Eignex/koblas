package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.SparseMatrix
import kotlin.math.abs

/** Only stored rows need accumulation, even when the logical row domain is too large for a dense buffer. */
internal fun sparseMatrixInfinityNorm(a: SparseMatrix): Double {
    val sums = mutableMapOf<Int, Double>()
    for (j in 0 until a.cols) {
        a.forEachInColumn(j) { i, value -> sums[i] = (sums[i] ?: 0.0) + abs(value) }
    }
    var result = 0.0
    for (sum in sums.values) result = maxOf(result, sum)
    return result
}
