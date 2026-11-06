package com.eignex.koblas.internal.kernels.scalar.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.Vector

/** Merge sparse supports without scanning the logical domain or evaluating absent products. */
internal fun scalarSparseVectorAxpy(alpha: Double, x: Vector, y: SparseVector): SparseVector {
    val indices = mutableListOf<Int>()
    val values = mutableListOf<Double>()
    var b = 0
    if (x is SparseVector) {
        var a = 0
        while (a < x.nnz || b < y.nnz) {
            val ia = if (a < x.nnz) x.indices[a] else Int.MAX_VALUE
            val ib = if (b < y.nnz) y.indices[b] else Int.MAX_VALUE
            val index = minOf(ia, ib)
            val previous = if (ib == index) y.values[b++] else 0.0
            val value = if (ia == index) previous + alpha * x.values[a++] else previous
            indices += index
            values += value
        }
    } else {
        for (i in 0 until x.size) {
            val stored = b < y.nnz && y.indices[b] == i
            val previous = if (stored) y.values[b++] else 0.0
            val value = previous + alpha * x[i]
            if (stored || value.toBits() != 0L) {
                indices += i
                values += value
            }
        }
    }
    return SparseVector.wrap(y.size, indices.toIntArray(), values.toDoubleArray())
}
