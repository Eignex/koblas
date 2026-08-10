package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/**
 * Both halves of the compute seam at once: the [Blas] routines and the [Lapack] factorizations built on
 * them. [koblas] is one of these, composed from whichever backend won each half.
 *
 * Implement this when a backend provides both, which is the usual case for a host library. Implement
 * [Blas] or [Lapack] alone when it does not — the two are ranked and installed independently, so a host
 * with CBLAS but no LAPACKE still accelerates its level-2 and level-3 work.
 */
public interface LinearAlgebra :
    Blas,
    Lapack

// Every dense factorization is reached the same way: a verb on the matrix produces the decomposition, and
// the solves hang off the decomposition. One shape to learn rather than one per family, and the same shape
// the sparse side already had.

/** LU-factorize this square matrix with the active backend ([koblas]); see [Lapack.factor]. */
public fun DenseMatrix.lu(): LuDecomposition = koblas.factor(this)

/** Symmetric indefinite factorization `A = L·D·Lᵀ` with the active backend; see [Lapack.ldl]. */
public fun DenseMatrix.ldl(workspace: Workspace? = null): LdlDecomposition = koblas.ldl(this, workspace)

/** QR factorization `A = Q·R` with the active backend; see [Lapack.qr]. */
public fun DenseMatrix.qr(workspace: Workspace? = null): QrDecomposition = koblas.qr(this, workspace)

/**
 * QR with column pivoting, `A·P = Q·R`, with the active backend — the factorization that reports a
 * numerical [PivotedQrDecomposition.rank]; see [Lapack.qrPivoted].
 */
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
 * Reciprocal condition estimate for this factorization, given the 1-norm [anorm] of the matrix it came
 * from; see [Lapack.rcond]. Pair it with [com.eignex.koblas.norm1], computed before factoring.
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
