@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
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

/**
 * Allocates a product using this engine. Two sparse operands produce owned CSC storage;
 * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
 */
public fun KoblasEngine.gemm(a: Matrix, b: Matrix): Matrix = gemm(1.0, a, false, b, false)

/** Scaled product with independent transpose flags and the storage rule of [gemm]. */
public fun KoblasEngine.gemm(alpha: Double, a: Matrix, transposeA: Boolean, b: Matrix, transposeB: Boolean): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    if (a is SparseMatrix && b is SparseMatrix) return sparseGemm(alpha, a, transposeA, b, transposeB)
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    gemm(alpha, a, transposeA, b, transposeB, 0.0, result)
    return result
}

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
) {
    requireGemmOperands(a, transposeA, b, transposeB, c)
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    when {
        a is SparseMatrix && b is SparseMatrix -> sparseGemm(alpha, a, transposeA, b, transposeB, beta, c)
        a is SparseMatrix -> denseOperand(arrays, b) { right ->
            sparseGemm(alpha, a, transposeA, right, transposeB, beta, c, right = false)
        }
        b is SparseMatrix -> denseOperand(arrays, a) { left ->
            sparseGemm(alpha, b, transposeB, left, transposeA, beta, c, right = true)
        }
        else -> denseOperand(arrays, a) { left ->
            if (a === b) {
                denseGemm(alpha, left, transposeA, left, transposeB, beta, c)
            } else {
                denseOperand(arrays, b) { right -> denseGemm(alpha, left, transposeA, right, transposeB, beta, c) }
            }
        }
    }
}

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
    if (a is SparseMatrix) {
        sparseTrsv(a, x, lower, transpose, unitDiag)
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
    if (a is SparseMatrix) {
        sparseTrmv(a, x, lower, transpose, unitDiag)
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
        scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    if (a is SparseMatrix) {
        sparseTrsm(a, b, lower, transpose, unitDiag, right, alpha)
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
        scaleRegion(b, if (alpha == 0.0) 0.0 else 1.0)
        return
    }
    if (a is SparseMatrix) {
        sparseTrmm(a, b, lower, transpose, unitDiag, right, alpha)
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
