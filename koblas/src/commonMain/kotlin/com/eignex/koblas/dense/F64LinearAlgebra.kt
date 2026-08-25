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
 * Symmetric indefinite factorization `A = L·D·Lᵀ` with the active backend. [Uplo.FULL] checks that both
 * triangles agree; [Uplo.LOWER] or [Uplo.UPPER] selects one triangle without checking the other.
 */
public fun F64DenseMatrix.ldl(workspace: Workspace? = null, uplo: Uplo = Uplo.FULL): F64LdlDecomposition =
    koblas.ldl(asLowerSymmetricInput(uplo, "ldl"), workspace)

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
public fun F64LdlDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dsytrs` with `nrhs`). */
public fun F64LdlDecomposition.solve(b: F64DenseMatrix): F64DenseMatrix = koblas.solve(this, b)

/** Least-squares solution of `A · x ≈ b` for an overdetermined system; see [F64Decompositions.solveLeastSquares]. */
public fun F64QrDecomposition.solveLeastSquares(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveLeastSquares(this, b, workspace)

/** Minimum-norm solution of an underdetermined `A · x = b`; see [F64Decompositions.solveMinimumNorm]. */
public fun F64QrDecomposition.solveMinimumNorm(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveMinimumNorm(this, b, workspace)

/** `Q · y`, or `Qᵀ · y` when [transpose], without forming `Q`; see [F64Decompositions.applyQ]. */
public fun F64QrDecomposition.applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.applyQ(
    this,
    y,
    transpose,
)

/** Least-squares solution against a rank-revealing factorization; see [F64Decompositions.solveLeastSquares]. */
public fun F64PivotedQrDecomposition.solveLeastSquares(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveLeastSquares(this, b, workspace)
