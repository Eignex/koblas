package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.forEachStored
import kotlin.math.abs

/** Only stored rows need accumulation, even when the logical row domain is too large for a dense buffer. */
internal fun sparseMatrixInfinityNorm(a: SparseMatrix): Double {
    val sums = mutableMapOf<Int, Double>()
    a.forEachStored { i, _, value -> sums[i] = (sums[i] ?: 0.0) + abs(value) }
    var result = 0.0
    for (sum in sums.values) result = maxOf(result, sum)
    return result
}
