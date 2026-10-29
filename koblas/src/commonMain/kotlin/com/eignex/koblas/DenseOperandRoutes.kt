@file:Suppress("LongParameterList") // operand-based BLAS diagnostics mirror execution signatures
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.operands.regionVectorOverlap
import com.eignex.koblas.internal.operands.regionsOverlap

/** Matrix-vector route derived from actual regions, strides and aliases, without reading coefficients. */
public fun KoblasEngine.routeOf(
    operation: DenseMatrixOperation,
    alpha: Double,
    a: DenseMatrixRegion,
    x: DenseVector,
    beta: Double,
    y: DenseVector,
    lower: Boolean = true,
): DenseMatrixRoute {
    when (operation) {
        DenseMatrixOperation.Gemv -> requireGemvOperands(a, false, x.size, y.size)
        DenseMatrixOperation.GemvTransposed -> requireGemvOperands(a, true, x.size, y.size)
        DenseMatrixOperation.Symv -> requireSymvOperands(a, x.size, y.size)
        else -> error("$operation is not a matrix-vector product")
    }
    val overlaps = x.overlaps(y)
    val xContiguous = overlaps || x.stride == 1
    val contiguous = when (operation) {
        DenseMatrixOperation.Gemv -> y.stride == 1
        DenseMatrixOperation.GemvTransposed -> xContiguous
        else -> xContiguous && y.stride == 1
    }
    return routeOf(
        operation,
        DenseCall(
            a.rows,
            a.cols,
            alpha,
            beta,
            contiguous,
            lower = lower,
            aliased = overlaps || regionVectorOverlap(a, y),
            borrowed = a is DenseMatrixView,
        ),
    )
}

/** Rank-update route derived from the selected matrix region and both addressed vector runs. */
public fun KoblasEngine.routeOf(
    operation: DenseMatrixOperation,
    alpha: Double,
    x: DenseVector,
    y: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
): DenseMatrixRoute {
    when (operation) {
        DenseMatrixOperation.Ger, DenseMatrixOperation.GerWindow -> requireGerOperands(x.size, y.size, a)
        DenseMatrixOperation.Syr2 -> requireSyr2Operands(a, x.size, y.size, "syr2")
        else -> error("$operation is not a two-vector rank update")
    }
    val aliases = regionVectorOverlap(a, x) || regionVectorOverlap(a, y)
    return routeOf(
        operation,
        DenseCall(
            a.rows,
            a.cols,
            alpha,
            contiguous = x.stride == 1 || regionVectorOverlap(a, x),
            lower = lower,
            aliased = aliases,
            borrowed = a is DenseMatrixView,
        ),
    )
}

/** Symmetric rank-one route derived from the addressed vector and destination region. */
public fun KoblasEngine.routeOf(
    alpha: Double,
    x: DenseVector,
    a: DenseMatrixRegion,
    lower: Boolean = true,
): DenseMatrixRoute {
    requireSyrOperands(a, x.size, "syr")
    val aliases = regionVectorOverlap(a, x)
    return routeOf(
        DenseMatrixOperation.Syr,
        DenseCall(
            a.rows,
            a.cols,
            alpha,
            contiguous = x.stride == 1 || aliases,
            lower = lower,
            aliased = aliases,
            borrowed = a is DenseMatrixView,
        ),
    )
}

/** Triangular vector route; the operation selects multiplication/solve and stored/transposed traversal. */
public fun KoblasEngine.routeOf(
    operation: DenseMatrixOperation,
    a: DenseMatrixRegion,
    x: DenseVector,
    lower: Boolean = true,
): DenseMatrixRoute {
    require(
        operation in setOf(
            DenseMatrixOperation.Trsv,
            DenseMatrixOperation.TrsvTransposed,
            DenseMatrixOperation.Trmv,
            DenseMatrixOperation.TrmvTransposed,
        ),
    ) { "$operation is not triangular vector work" }
    requireTriangularVectorOperands(a, x.size, operation.name)
    return routeOf(
        operation,
        DenseCall(
            a.rows,
            a.cols,
            contiguous = x.stride == 1,
            lower = lower,
            aliased = regionVectorOverlap(a, x),
            borrowed = a is DenseMatrixView,
        ),
    )
}

/** Structured product route derived from the actual operand regions and destination spacing. */
public fun KoblasEngine.routeOf(
    operation: DenseMatrixOperation,
    alpha: Double,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    transposeA: Boolean,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
    right: Boolean = false,
): DenseMatrixRoute {
    when (operation) {
        DenseMatrixOperation.Gemm -> requireGemmOperands(a, transposeA, b, transposeB, c)
        DenseMatrixOperation.Gemmt -> requireGemmtOperands(a, transposeA, b, transposeB, c)
        DenseMatrixOperation.Symm -> requireSymmOperands(a, b, c, right)
        DenseMatrixOperation.Syr2k -> requireSyr2kOperands(a, b, transposeA, c)
        else -> error("$operation is not a two-operand matrix product")
    }
    val depth = if (operation == DenseMatrixOperation.Symm || transposeA) a.rows else a.cols
    return routeOf(
        operation,
        DenseCall(
            c.rows, c.cols, alpha, beta, depth = depth, lower = lower,
            transposeA = transposeA, transposeB = transposeB, right = right,
            aliased = regionsOverlap(a, c) || regionsOverlap(b, c),
            borrowed = a is DenseMatrixView || b is DenseMatrixView || c is DenseMatrixView,
            leadingDimension = c.leadingDimension, rightLeadingDimension = b.leadingDimension,
            leftLeadingDimension = a.leadingDimension,
        ),
    )
}

/** Symmetric rank-k route derived from operand and selected destination regions. */
public fun KoblasEngine.routeOf(
    alpha: Double,
    a: DenseMatrixRegion,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    lower: Boolean = true,
): DenseMatrixRoute {
    requireSyrkOperands(a, transpose, c)
    return routeOf(
        DenseMatrixOperation.Syrk,
        DenseCall(
            c.rows, c.cols, alpha, beta,
            depth = if (transpose) a.rows else a.cols, lower = lower, transposeA = transpose,
            aliased = regionsOverlap(a, c), borrowed = a is DenseMatrixView || c is DenseMatrixView,
            leadingDimension = c.leadingDimension, leftLeadingDimension = a.leadingDimension,
        ),
    )
}

/** Triangular matrix route derived from the triangle and the in-place right-hand-side region. */
public fun KoblasEngine.routeOf(
    operation: DenseMatrixOperation,
    a: DenseMatrixRegion,
    b: DenseMatrixRegion,
    lower: Boolean = true,
    transpose: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): DenseMatrixRoute {
    require(operation == DenseMatrixOperation.Trsm || operation == DenseMatrixOperation.Trmm)
    requireTriangularMatrixOperands(a, b, right, operation.name)
    return routeOf(
        operation,
        DenseCall(
            b.rows, b.cols, alpha, depth = a.rows,
            lower = lower, transposeA = transpose, right = right, aliased = regionsOverlap(a, b),
            borrowed = a is DenseMatrixView || b is DenseMatrixView, leadingDimension = b.leadingDimension,
            leftLeadingDimension = a.leadingDimension,
        ),
    )
}
