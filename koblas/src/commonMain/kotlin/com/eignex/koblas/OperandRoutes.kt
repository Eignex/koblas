@file:Suppress("LongParameterList") // the matrix product signature
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.RouteKind
import com.eignex.koblas.internal.diagnostics.productRoute
import com.eignex.koblas.internal.diagnostics.sharesBuffer

/**
 * Describes a [Matrix.gemmInto] product from its operands, without reading values or preparing storage.
 * Execution and inspection both use this receiver engine, including prepared operands.
 */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
): MatrixRoute {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    return productRoute(
        alpha, a, transposeA, b, transposeB, beta, c.rows, c.cols, DenseMatrix.entryCount(c.rows, c.cols),
        aliased = sharesBuffer(a, c) || sharesBuffer(b, c), destination = c,
    )
}

/**
 * Describes [Matrix.gemm] without allocating its result. Two sparse operands, prepared or ordinary,
 * produce CSC storage; other pairings produce dense storage. No cached orientation is constructed.
 */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): MatrixRoute {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    val rows = if (transposeA) a.cols else a.rows
    val columns = if (transposeB) b.rows else b.cols
    val sparsePair = storedCsc(a) != null && storedCsc(b) != null
    return productRoute(
        alpha, a, transposeA, b, transposeB, 0.0, rows, columns,
        elements = if (sparsePair) null else DenseMatrix.entryCount(rows, columns), aliased = false,
    )
}
