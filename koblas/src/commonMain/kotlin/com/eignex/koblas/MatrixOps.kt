@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import kotlin.jvm.JvmOverloads

/**
 * Allocates a dense product using this engine, with at most one sparse operand.
 * Custom [Matrix] operands are read once into dense scratch.
 */
@JvmOverloads
public fun KoblasEngine.gemm(a: Matrix, b: Matrix, scratch: Scratch? = null): Matrix =
    gemm(1.0, a, false, b, false, scratch)

/** Scaled product with independent transpose flags and the storage rule of [gemm]. */
@JvmOverloads
public fun KoblasEngine.gemm(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    scratch: Scratch? = null,
): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    require(a !is SparseMatrix || b !is SparseMatrix) { "gemm requires at least one dense operand" }
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    gemm(alpha, a, transposeA, b, transposeB, 0.0, result, scratch)
    return result
}

/**
 * Writes a product of readable matrices, with at most one sparse operand, into addressed dense storage.
 * Validation precedes all writes; zero alpha and empty depth read no inputs, and zero beta reads no destination values.
 * A custom [Matrix] operand is read once into dense storage borrowed from [scratch] when one is given.
 */
@JvmOverloads
public fun KoblasEngine.gemm(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
    scratch: Scratch? = null,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    require(a !is SparseMatrix || b !is SparseMatrix) { "gemm requires at least one dense operand" }
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    when {
        a is SparseMatrix -> denseOperand(scratch, b) { right ->
            gemm(alpha, a, transposeA, right, transposeB, beta, c, right = false, scratch = scratch)
        }

        b is SparseMatrix -> denseOperand(scratch, a) { left ->
            gemm(alpha, b, transposeB, left, transposeA, beta, c, right = true, scratch = scratch)
        }

        else -> denseOperand(scratch, a) { left ->
            if (a === b) {
                gemm(alpha, left, transposeA, left, transposeB, beta, c, scratch)
            } else {
                denseOperand(scratch, b) { right -> gemm(alpha, left, transposeA, right, transposeB, beta, c, scratch) }
            }
        }
    }
}

/** TRSV over readable dense, sparse or custom operands, using this engine. */
@JvmOverloads
public fun KoblasEngine.trsv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    withTriangle(a, lower, unitDiag, scratch, { trsv(it, x, lower, transpose, unitDiag, scratch) }) {
        trsv(it, x, 0, 1, lower, transpose, unitDiag, scratch)
    }
}

/** TRMV over readable dense, sparse or custom operands, using this engine. */
@JvmOverloads
public fun KoblasEngine.trmv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    withTriangle(a, lower, unitDiag, scratch, { trmv(it, x, lower, transpose, unitDiag, scratch) }) {
        trmv(it, x, 0, 1, lower, transpose, unitDiag, scratch)
    }
}

/** TRSM over readable dense, sparse or custom operands, using this engine. */
@JvmOverloads
public fun KoblasEngine.trsm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    scratch: Scratch? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    withTriangle(a, lower, unitDiag, scratch, { trsm(it, b, lower, transpose, unitDiag, right, alpha, scratch) }) {
        trsm(it, b, lower, transpose, unitDiag, right, alpha, scratch)
    }
}

/** TRMM over readable dense, sparse or custom operands, using this engine. */
@JvmOverloads
public fun KoblasEngine.trmm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    scratch: Scratch? = null,
) {
    requireTriangularMatrixOperands(a, b, right, "trmm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    withTriangle(a, lower, unitDiag, scratch, { trmm(it, b, lower, transpose, unitDiag, right, alpha, scratch) }) {
        trmm(it, b, lower, transpose, unitDiag, right, alpha, scratch)
    }
}

/** A sparse triangle keeps its structure; any other operand is read once into dense triangle storage. */
private inline fun withTriangle(
    a: Matrix,
    lower: Boolean,
    unitDiag: Boolean,
    scratch: Scratch?,
    sparse: (SparseMatrix) -> Unit,
    dense: (DenseMatrixRegion) -> Unit,
) {
    if (a is SparseMatrix) sparse(a) else triangleOperand(scratch, a, lower, unitDiag, dense)
}
