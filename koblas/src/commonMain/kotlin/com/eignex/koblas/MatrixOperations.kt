@file:Suppress("LongParameterList") // the BLAS signature and product family
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*

/**
 * Allocates a product using this engine. Two sparse operands produce owned CSC storage;
 * other pairings produce dense storage. Custom [Matrix] operands are read once into dense scratch.
 */
public fun KoblasEngine.gemm(a: Matrix, b: Matrix, scratch: Scratch? = null): Matrix =
    gemm(1.0, a, false, b, false, scratch)

/** Scaled product with independent transpose flags and the storage rule of [gemm]. */
public fun KoblasEngine.gemm(
    alpha: Double,
    a: Matrix,
    transposeA: Boolean,
    b: Matrix,
    transposeB: Boolean,
    scratch: Scratch? = null,
): Matrix {
    requireProductOperands(a, transposeA, b, transposeB, "gemm")
    if (a is SparseMatrix && b is SparseMatrix) return gemm(alpha, a, transposeA, b, transposeB)
    val result = DenseMatrix.zero(if (transposeA) a.cols else a.rows, if (transposeB) b.rows else b.cols)
    gemm(alpha, a, transposeA, b, transposeB, 0.0, result, scratch)
    return result
}

/**
 * Writes a product of any two readable matrices through this engine into addressed dense storage. Validation
 * precedes all writes; zero alpha and empty depth read no inputs, and zero beta reads no destination values.
 * A custom [Matrix] operand is read once into dense storage borrowed from [scratch] when one is given.
 */
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
    if (c.rows == 0 || c.cols == 0) return
    if (alpha == 0.0 || (if (transposeA) a.rows else a.cols) == 0) {
        scaleRegion(c, beta, vectors = this)
        return
    }
    val arrays = scratch?.arrays
    when {
        a is SparseMatrix && b is SparseMatrix -> gemm(alpha, a, transposeA, b, transposeB, beta, c, scratch)

        a is SparseMatrix -> denseOperand(arrays, b) { right ->
            gemm(alpha, a, transposeA, right, transposeB, beta, c, right = false, scratch = scratch)
        }

        b is SparseMatrix -> denseOperand(arrays, a) { left ->
            gemm(alpha, b, transposeB, left, transposeA, beta, c, right = true, scratch = scratch)
        }

        else -> denseOperand(arrays, a) { left ->
            if (a === b) {
                gemm(alpha, left, transposeA, left, transposeB, beta, c, scratch)
            } else {
                denseOperand(arrays, b) { right -> gemm(alpha, left, transposeA, right, transposeB, beta, c, scratch) }
            }
        }
    }
}

/** TRSV over readable dense, sparse or custom operands, using this engine. */
public fun KoblasEngine.trsv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trsv")
    if (a is SparseMatrix) {
        trsv(a, x, lower, transpose, unitDiag, scratch)
    } else {
        triangleOperand(scratch?.arrays, a, lower, unitDiag) { dense ->
            trsv(dense, x, 0, 1, lower, transpose, unitDiag, scratch)
        }
    }
}

/** TRMV over readable dense, sparse or custom operands, using this engine. */
public fun KoblasEngine.trmv(
    a: Matrix,
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    scratch: Scratch? = null,
) {
    requireTriangularVectorOperands(a, x.size, "trmv")
    if (a is SparseMatrix) {
        trmv(a, x, lower, transpose, unitDiag, scratch)
    } else {
        triangleOperand(scratch?.arrays, a, lower, unitDiag) { dense ->
            trmv(dense, x, 0, 1, lower, transpose, unitDiag, scratch)
        }
    }
}

/** TRSM over readable dense, sparse or custom operands, using this engine. */
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
    if (a is SparseMatrix) {
        trsm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
    } else {
        triangleOperand(scratch?.arrays, a, lower, unitDiag) { dense ->
            trsm(dense, b, lower, transpose, unitDiag, right, alpha, scratch)
        }
    }
}

/** TRMM over readable dense, sparse or custom operands, using this engine. */
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
    if (a is SparseMatrix) {
        trmm(a, b, lower, transpose, unitDiag, right, alpha, scratch)
    } else {
        triangleOperand(scratch?.arrays, a, lower, unitDiag) { dense ->
            trmm(dense, b, lower, transpose, unitDiag, right, alpha, scratch)
        }
    }
}

/** Allocating product through [koblas]: two sparse operands give CSC storage, any other pairing dense. */
@kotlin.jvm.JvmName("multiplyMatrices")
public operator fun Matrix.times(other: Matrix): Matrix = koblas.gemm(this, other)
