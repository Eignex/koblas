@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: L, Q, R, U

package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas

/**
 * A standalone lower-trapezoidal `L` with unit diagonal from this `P·A = L·U` factorization.
 *
 * The returned matrix is a copy: changing it cannot change this factorization or later solves.
 */
public fun F64LuDecomposition.lowerFactor(): F64DenseMatrix = lower()

/**
 * A standalone upper-trapezoidal `U` from this `P·A = L·U` factorization.
 *
 * The returned matrix is a copy: changing it cannot change this factorization or later solves.
 */
public fun F64LuDecomposition.upperFactor(): F64DenseMatrix = upper()

/**
 * A defensive copy of the row permutation, where entry `k` is the original row at pivot position `k`.
 */
public fun F64LuDecomposition.rowOrder(): IntArray = rowPermutation

/**
 * The sign of `det(A)`, including the row permutation. It is zero when this factorization is singular or
 * when a diagonal entry is `NaN`, since neither has a well-defined sign.
 */
public fun F64LuDecomposition.determinantSign(): Int {
    requireLuSquare(this, "determinantSign")
    if (singular) return 0
    var sign = if (permutationSign(mutablePivots) > 0.0) 1 else -1
    for (k in 0 until n) {
        val diagonal = lu[k + k * n]
        when {
            diagonal.isNaN() || diagonal == 0.0 -> return 0
            diagonal < 0.0 -> sign = -sign
        }
    }
    return sign
}

/**
 * `log(abs(det(A)))`, accumulated without the overflow or underflow of [determinant].
 *
 * A singular factorization returns [Double.NEGATIVE_INFINITY]. Pair this with [determinantSign] when the
 * determinant's sign matters.
 */
public fun F64LuDecomposition.logAbsDeterminant(): Double {
    requireLuSquare(this, "logAbsDeterminant")
    if (singular) return Double.NEGATIVE_INFINITY
    var logAbs = 0.0
    for (k in 0 until n) {
        val diagonal = lu[k + k * n]
        if (diagonal == 0.0) return Double.NEGATIVE_INFINITY
        logAbs += kotlin.math.ln(kotlin.math.abs(diagonal))
    }
    return logAbs
}

/**
 * A standalone lower-triangular `L` from this Cholesky factorization `A = L·Lᵀ`.
 *
 * The returned matrix is a copy and its strict upper triangle is zeroed.
 */
public fun F64CholeskyDecomposition.lowerFactor(): F64DenseMatrix = F64DenseMatrix(n, n).also { lower ->
    for (column in 0 until n) for (row in column until n) lower[row, column] = l[row, column]
}

/**
 * A defensive copy of the packed lower-triangular LAPACK `dsytrf` data.
 *
 * Bunch-Kaufman interleaves the `D` blocks, multipliers, and pivoting in this representation, so it has no
 * independent conventional `L` and diagonal `D` matrix to expose without changing its numerical contract.
 * Unlike [F64LuDecomposition.lowerFactor] or [F64CholeskyDecomposition.lowerFactor], the strict upper
 * triangle here is not zeroed: `dsytrf` never writes above the diagonal, so those entries are whatever the
 * factored matrix held there, not factorization data.
 */
public fun F64PivotedSymmetricIndefiniteDecomposition.packedFactor(): F64DenseMatrix =
    F64DenseMatrix(n, n, ldl.copyOf())

/**
 * The explicit orthogonal `m×m` matrix `Q` represented by this factorization.
 *
 * This applies the backend's existing `Q` operator to identity columns; it does not reimplement the
 * Householder kernel. Supplying [workspace] reuses two length-`m` staging vectors across all columns.
 */
public fun F64QrDecomposition.explicitQ(workspace: Workspace? = null): F64DenseMatrix {
    val q = F64DenseMatrix.diagonal(m)
    workspace.borrow(m) { input ->
        workspace.borrow(m) { output ->
            for (column in 0 until m) {
                input.fill(0.0)
                input[column] = 1.0
                koblas.applyQInto(this, input, output)
                output.copyInto(q.data, column * m)
            }
        }
    }
    return q
}

/**
 * The explicit `m×n` upper-trapezoidal `R` from this `A = Q·R` factorization.
 *
 * The returned matrix is a copy and has zeros below the conventional `R` triangle.
 */
public fun F64QrDecomposition.explicitR(): F64DenseMatrix = F64DenseMatrix(m, n).also { r ->
    val diagonalLength = minOf(m, n)
    for (column in 0 until n) {
        for (row in 0 until minOf(column + 1, diagonalLength)) r[row, column] = qr[row + column * m]
    }
}

/** A defensive copy of the column order, where entry `k` is the original column at pivot position `k`. */
public fun F64PivotedQrDecomposition.columnOrder(): IntArray = pivots.copyOf()

/** The explicit `Q` of this pivoted factorization; [explicitR] reconstructs `A·P = Q·R`. */
public fun F64PivotedQrDecomposition.explicitQ(workspace: Workspace? = null): F64DenseMatrix =
    factorization.explicitQ(workspace)

/** The explicit `R` of this pivoted factorization; [columnOrder] supplies its `P`. */
public fun F64PivotedQrDecomposition.explicitR(): F64DenseMatrix = factorization.explicitR()
