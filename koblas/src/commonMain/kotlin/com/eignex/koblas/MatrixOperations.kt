@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.operands.denseOperand
import com.eignex.koblas.internal.operands.scaleRegion
import com.eignex.koblas.internal.operands.triangleOperand
import com.eignex.koblas.sparse.*
import com.eignex.koblas.dense.gemm as denseGemm
import com.eignex.koblas.dense.trmm as denseTrmm
import com.eignex.koblas.dense.trmv as denseTrmv
import com.eignex.koblas.dense.trsm as denseTrsm
import com.eignex.koblas.dense.trsv as denseTrsv
import com.eignex.koblas.sparse.gemm as sparseGemm
import com.eignex.koblas.sparse.trmm as sparseTrmm
import com.eignex.koblas.sparse.trmv as sparseTrmv
import com.eignex.koblas.sparse.trsm as sparseTrsm
import com.eignex.koblas.sparse.trsv as sparseTrsv

// Whole-call hooks are reached only after capability selection, validation and alias staging.

/**
 * Allocates a product using this engine. Two sparse operands produce owned CSC storage;
 * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
 */
public fun KoblasEngine.gemm(a: Matrix, b: Matrix): Matrix = product(1.0, a, false, b, false)

/** Scaled product with independent transpose flags and the storage rule of [gemm]. */
public fun KoblasEngine.gemm(alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean): Matrix =
    product(alpha, a, transposeA, b, transposeB)

/**
 * Writes a generic product through this engine into addressed dense storage. Validation precedes all
 * writes; zero alpha and empty depth read no inputs, and zero beta reads no destination values.
 * Reusable scratch supplies logical alias snapshots, sparse scratch and custom operand normalization.
 */
@Suppress("LongParameterList") // BLAS operands, flags and scratch
public fun KoblasEngine.gemm(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
): Unit = productInto(alpha, a, transposeA, b, transposeB, beta, c)

/** TRSV over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and scratch
public fun KoblasEngine.trsv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    val sparse = storedCsc(a)
    if (sparse != null) {
        sparseTrsv(sparse, x, lower, transpose, unitDiag)
    } else {
        triangleOperand(
            arrays,
            a,
            lower,
            unitDiag,
        ) { dense -> denseTrsv(dense, x, lower, transpose, unitDiag) }
    }
}

/** TRMV over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and scratch
public fun KoblasEngine.trmv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    val sparse = storedCsc(a)
    if (sparse != null) {
        sparseTrmv(sparse, x, lower, transpose, unitDiag)
    } else {
        triangleOperand(
            arrays,
            a,
            lower,
            unitDiag,
        ) { dense -> denseTrmv(dense, x, lower, transpose, unitDiag) }
    }
}

/** TRSM over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and scratch
public fun KoblasEngine.trsm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
) {
    requireTriangularMatrixOperands(a, b, right, "trsm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    val sparse = storedCsc(a)
    if (sparse != null) {
        sparseTrsm(sparse, b, lower, transpose, unitDiag, right, alpha)
    } else {
        triangleOperand(
            arrays,
            a,
            lower,
            unitDiag,
        ) { dense -> denseTrsm(dense, b, lower, transpose, unitDiag, right, alpha) }
    }
}

/** TRMM over readable dense, sparse or custom operands, using this engine. */
@Suppress("LongParameterList") // triangle flags and scratch
public fun KoblasEngine.trmm(
    a: Matrix,
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
) {
    requireTriangularMatrixOperands(a, b, right, "trmm")
    if (alpha == 0.0 || b.rows == 0 || b.cols == 0) {
        com.eignex.koblas.internal.operands.scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    val sparse = storedCsc(a)
    if (sparse != null) {
        sparseTrmm(sparse, b, lower, transpose, unitDiag, right, alpha)
    } else {
        triangleOperand(
            arrays,
            a,
            lower,
            unitDiag,
        ) { dense -> denseTrmm(dense, b, lower, transpose, unitDiag, right, alpha) }
    }
}

/** Allocating product through [koblas]: two sparse operands give CSC storage, any other pairing dense. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = koblas.gemm(this, other)

/** Storage classification; callbacks inline on the arithmetic path. */
internal inline fun <T> dispatchProduct(
    a: Matrix,
    b: Matrix,
    sparsePair: (SparseMatrix, SparseMatrix) -> T,
    sparseLeft: (SparseMatrix) -> T,
    sparseRight: (SparseMatrix) -> T,
    densePair: () -> T,
): T {
    val left = storedCsc(a)
    val right = storedCsc(b)
    return when {
        left != null && right != null -> sparsePair(left, right)
        left != null -> sparseLeft(left)
        right != null -> sparseRight(right)
        else -> densePair()
    }
}

/** Engine-owned allocating dispatch, with no policy retained by either operand. */
internal fun KoblasEngine.product(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    return dispatchProduct(
        a,
        b,
        sparsePair = { left, right -> sparseGemm(alpha, left, transposeA, right, transposeB) },
        sparseLeft = { denseProduct(alpha, a, transposeA, b, transposeB) },
        sparseRight = { denseProduct(alpha, a, transposeA, b, transposeB) },
        densePair = { denseProduct(alpha, a, transposeA, b, transposeB) },
    )
}

private fun KoblasEngine.denseProduct(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
): DenseMatrix {
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    productInto(alpha, a, transposeA, b, transposeB, 0.0, result)
    return result
}

/** Canonical generic destination dispatch. Sparse structure never passes through custom materialization. */
internal fun KoblasEngine.productInto(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    beta: Double,
    c: DenseMatrixRegion,
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    dispatchProduct(
        a,
        b,
        sparsePair = { left, right -> sparseGemm(alpha, left, transposeA, right, transposeB, beta, c) },
        sparseLeft = { left ->
            denseOperand(arrays, b) { right ->
                sparseGemm(alpha, left, transposeA, right, transposeB, beta, c, right = false)
            }
        },
        sparseRight = { right ->
            denseOperand(arrays, a) { left ->
                sparseGemm(alpha, right, transposeB, left, transposeA, beta, c, right = true)
            }
        },
        densePair = {
            denseOperand(arrays, a) { left ->
                if (a === b) {
                    denseGemm(alpha, left, transposeA, left, transposeB, beta, c)
                } else {
                    denseOperand(
                        arrays,
                        b,
                    ) { right -> denseGemm(alpha, left, transposeA, right, transposeB, beta, c) }
                }
            }
        },
    )
}

/** The CSC an operand traverses, or null when it has none. */
internal fun storedCsc(matrix: Matrix): SparseMatrix? = matrix as? SparseMatrix
