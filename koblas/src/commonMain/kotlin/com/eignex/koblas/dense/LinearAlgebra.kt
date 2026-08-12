package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/** Both halves of the compute seam at once; implement [Blas] or [Lapack] alone when a backend has one. */
public interface LinearAlgebra :
    Blas,
    Lapack

/** LU-factorize this square matrix with the active backend ([koblas]); see [Lapack.factor]. */
public fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Symmetric indefinite factorization `A = L·D·Lᵀ` with the active backend; see [Lapack.ldl]. */
public fun DenseMatrix.ldl(workspace: Workspace? = null): LdlDecomposition = koblas.ldl(this, workspace)

/** QR factorization `A = Q·R` with the active backend; see [Lapack.qr]. */
public fun DenseMatrix.qr(workspace: Workspace? = null): QrDecomposition = koblas.qr(this, workspace)

/** QR with column pivoting, `A·P = Q·R`, with the active backend; see [Lapack.qrPivoted]. */
public fun DenseMatrix.qrPivoted(
    tolerance: Double = AUTOMATIC_RANK_TOLERANCE,
    workspace: Workspace? = null,
): PivotedQrDecomposition = koblas.qrPivoted(this, tolerance, workspace)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
public fun LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(
    this,
    b,
    transpose,
)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dgetrs` with `nrhs`). */
public fun LuDecomposition.solve(b: DenseMatrix, transpose: Boolean = false): DenseMatrix = koblas.solve(
    this,
    b,
    transpose,
)

/** `A⁻¹` from this factorization (LAPACK `dgetri`); see [Lapack.invert]. */
public fun LuDecomposition.invert(workspace: Workspace? = null): DenseMatrix = koblas.invert(this, workspace)

/**
 * Reciprocal condition estimate, given the 1-norm [anorm] of the matrix it came from; see [Lapack.rcond].
 * Pair it with [com.eignex.koblas.norm1], computed before factoring.
 */
public fun LuDecomposition.rcond(anorm: Double, workspace: Workspace? = null): Double = koblas.rcond(
    this,
    anorm,
    workspace,
)

/** Solve `A · x = b` for this symmetric indefinite factorization; see [Lapack.solve]. */
public fun LdlDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dsytrs` with `nrhs`). */
public fun LdlDecomposition.solve(b: DenseMatrix): DenseMatrix = koblas.solve(this, b)

/** Least-squares solution of `A · x ≈ b` for an overdetermined system; see [Lapack.solveLeastSquares]. */
public fun QrDecomposition.solveLeastSquares(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveLeastSquares(this, b, workspace)

/** Minimum-norm solution of an underdetermined `A · x = b`; see [Lapack.solveMinimumNorm]. */
public fun QrDecomposition.solveMinimumNorm(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveMinimumNorm(this, b, workspace)

/** `Q · y`, or `Qᵀ · y` when [transpose], without forming `Q`; see [Lapack.applyQ]. */
public fun QrDecomposition.applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.applyQ(
    this,
    y,
    transpose,
)

/** Least-squares solution against a rank-revealing factorization; see [Lapack.solveLeastSquares]. */
public fun PivotedQrDecomposition.solveLeastSquares(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solveLeastSquares(this, b, workspace)

/** Matrix-matrix product `this · other` with the active backend. */
public fun DenseMatrix.matMul(other: DenseMatrix): DenseMatrix = koblas.gemm(this, other)
