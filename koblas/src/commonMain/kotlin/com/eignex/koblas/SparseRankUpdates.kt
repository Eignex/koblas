@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.internal.sparse.sparseSyr
import com.eignex.koblas.internal.sparse.sparseSyr2
import kotlin.jvm.JvmOverloads

/**
 * Fresh CSC matrix holding `A + alpha * x * xT` in its [lower] or upper triangle. The other triangle is
 * copied unchanged. Unlike dense [com.eignex.koblas.dense.DenseBlas.syr], this is not in place: a rank
 * update can introduce entries that the source CSC pattern has no room to store.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * arithmetic cancels or underflows to zero, so the returned matrix never silently drops discovered fill.
 * Zero vector entries are skipped, including with non-finite operands; dense NaN propagation is not guaranteed.
 * The result owns independent structural and value arrays, and its rows ascend within every column.
 */
@JvmOverloads
public fun SparseMatrix.syr(alpha: Double, x: Vector, lower: Boolean = true): SparseMatrix {
    requireSyrOperands(this, x.size, "syr")
    return sparseSyr(this, alpha, x, lower)
}

/**
 * Fresh CSC matrix holding `A + alpha * (x * yT + y * xT)` in its [lower] or upper triangle. The other
 * triangle is copied unchanged. This structural counterpart of dense
 * [com.eignex.koblas.dense.DenseBlas.syr2] never mutates its source, because newly nonzero entries may
 * require CSC fill.
 *
 * Existing explicit zeros survive. A coordinate reached by nonzero vector support is stored even when its
 * two terms cancel or underflow to zero. Zero vector entries are skipped as in [SparseMatrix.syr].
 * The result has independent arrays and canonical ascending CSC rows.
 */
@JvmOverloads
public fun SparseMatrix.syr2(alpha: Double, x: Vector, y: Vector, lower: Boolean = true): SparseMatrix {
    requireSyr2Operands(this, x.size, y.size, "syr2")
    return sparseSyr2(this, alpha, x, y, lower)
}
