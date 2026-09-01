package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas

/** Both halves of the compute seam at once; implement [F64Blas] or [F64Decompositions] alone when a backend has one. */
public interface F64LinearAlgebra :
    F64Blas,
    F64Decompositions {
    /** Resolves the two halves' declarations, so implementing both does not force a choice. */
    override val kernels: F64Kernels get() = koblas.kernels
}

/** LU-factorize this square matrix with the active backend ([koblas]); see [F64Decompositions.factor]. */
public fun F64DenseMatrix.lu(): F64LuDecomposition = koblas.factor(this)

/**
 * Symmetric indefinite factorization `A = L·D·Lᵀ` with the active backend. [lower] selects the authoritative
 * triangle without checking the other.
 */
public fun F64DenseMatrix.pivotedSymmetricIndefinite(
    workspace: Workspace? = null,
    lower: Boolean = true,
): F64PivotedSymmetricIndefiniteDecomposition = koblas.pivotedSymmetricIndefinite(
    asLowerSymmetricInput(lower, "pivotedSymmetricIndefinite"),
    workspace,
)

/** QR factorization `A = Q·R` with the active backend; see [F64Decompositions.qr]. */
public fun F64DenseMatrix.qr(workspace: Workspace? = null): F64QrDecomposition = koblas.qr(this, workspace)

/** QR with column pivoting, `A·P = Q·R`, with the active backend; see [F64Decompositions.qrPivoted]. */
public fun F64DenseMatrix.qrPivoted(
    tolerance: Double = AUTOMATIC_RANK_TOLERANCE,
    workspace: Workspace? = null,
): F64PivotedQrDecomposition = koblas.qrPivoted(this, tolerance, workspace)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
public fun F64LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(
    this,
    b,
    transpose,
)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dgetrs` with `nrhs`). */
public fun F64LuDecomposition.solve(b: F64DenseMatrix, transpose: Boolean = false): F64DenseMatrix = koblas.solve(
    this,
    b,
    transpose,
)

/** `A⁻¹` from this factorization (LAPACK `dgetri`); see [F64Decompositions.invert]. */
public fun F64LuDecomposition.invert(workspace: Workspace? = null): F64DenseMatrix = koblas.invert(this, workspace)

/**
 * Reciprocal condition estimate, given the 1-norm [anorm] of the matrix it came from; see [F64Decompositions.rcond].
 * Pair it with [com.eignex.koblas.norm1], computed before factoring.
 */
public fun F64LuDecomposition.rcond(anorm: Double, workspace: Workspace? = null): Double = koblas.rcond(
    this,
    anorm,
    workspace,
)

/** Solve `A · x = b` for this symmetric indefinite factorization; see [F64Decompositions.solve]. */
public fun F64PivotedSymmetricIndefiniteDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dsytrs` with `nrhs`). */
public fun F64PivotedSymmetricIndefiniteDecomposition.solve(b: F64DenseMatrix): F64DenseMatrix = koblas.solve(this, b)

/** Solve this QR factorization; [minimumNorm] solves a wide original system from `qr(Aᵀ)`. */
public fun F64QrDecomposition.solve(
    b: DoubleArray,
    minimumNorm: Boolean = false,
    workspace: Workspace? = null,
): DoubleArray = koblas.solve(this, b, minimumNorm, workspace)

/** `Q · y`, or `Qᵀ · y` when [transpose], without forming `Q`; see [F64Decompositions.applyQ]. */
public fun F64QrDecomposition.applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.applyQ(
    this,
    y,
    transpose,
)

/** Least-squares solution against this rank-revealing factorization; see [F64Decompositions.solve]. */
public fun F64PivotedQrDecomposition.solve(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solve(this, b, workspace)
