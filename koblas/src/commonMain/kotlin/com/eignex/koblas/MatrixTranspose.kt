@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.storage.sparse.transposeCsc

/**
 * Fresh transposed matrix. For products, prefer the transpose flags on [KoblasEngine.gemv] and
 * [KoblasEngine.gemm], which read the original storage without copying.
 */
public fun DenseMatrix.transpose(): DenseMatrix {
    val result = DenseMatrix.zero(cols, rows)
    for (j in 0 until cols) {
        for (i in 0 until rows) result.values[j + i * cols] = values[i + j * rows]
    }
    return result
}

/**
 * Fresh transposed matrix, still CSC, which makes this the CSC-to-CSR conversion as well. Explicitly stored
 * zeros survive, since the transpose is structural rather than arithmetic.
 */
public fun SparseMatrix.transpose(): SparseMatrix = transposeCsc(this)
