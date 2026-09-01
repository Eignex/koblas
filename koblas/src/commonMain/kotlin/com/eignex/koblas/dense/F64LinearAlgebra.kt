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

/** LU-factorize this matrix with the active backend ([koblas]); see [F64Decompositions.factor]. */
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

/**
 * Replaces this LU factorization with that of [a], retaining this decomposition's factor and pivot buffers.
 *
 * This is useful for a sequence of same-sized dense systems. Unlike a sparse symbolic analysis, a dense LU
 * has no pattern-owned state to preserve: the factor buffers are the whole reusable state. Implementations may
 * still need provider-local scratch, so this is a buffer-reuse contract rather than an allocation guarantee.
 */
public fun F64LuDecomposition.refactorInto(a: F64DenseMatrix): F64LuDecomposition = koblas.factorInto(a, this)

/** Solve `A · x = b` (or `Aᵀ · x = b` when [transpose]) for this factorization with the active backend. */
public fun F64LuDecomposition.solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.solve(
    this,
    b,
    transpose,
)

/**
 * Solve into [out], retaining the caller's destination. [out] may be [b]; pass [workspace] when the selected
 * backend needs staging, especially for an aliased or transposed solve.
 */
public fun F64LuDecomposition.solveInto(
    b: DoubleArray,
    out: DoubleArray,
    transpose: Boolean = false,
    workspace: Workspace? = null,
): DoubleArray = koblas.solveInto(this, b, out, transpose, workspace)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dgetrs` with `nrhs`). */
public fun F64LuDecomposition.solve(b: F64DenseMatrix, transpose: Boolean = false): F64DenseMatrix = koblas.solve(
    this,
    b,
    transpose,
)

/**
 * Solve every column of [b] into [out], retaining its backing buffer. [out] may be [b]; [workspace] supplies
 * any backend staging block, so reserve it before a hot loop when allocation behavior matters.
 */
public fun F64LuDecomposition.solveInto(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    transpose: Boolean = false,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solveInto(this, b, out, transpose, workspace)

/** `A⁻¹` from this factorization (LAPACK `dgetri`); see [F64Decompositions.invert]. */
public fun F64LuDecomposition.invert(workspace: Workspace? = null): F64DenseMatrix = koblas.invert(this, workspace)

/**
 * Reciprocal condition estimate, given the finite, non-negative 1-norm [anorm] of the matrix it came from;
 * see [F64Decompositions.rcond]. Pair it with [com.eignex.koblas.norm1], computed before factoring.
 */
public fun F64LuDecomposition.rcond(anorm: Double, workspace: Workspace? = null): Double = koblas.rcond(
    this,
    anorm,
    workspace,
)

/**
 * Replaces this LDL factorization with that of [a], retaining this decomposition's factor and pivot buffers.
 * [a]'s dimension must match this factorization's; implementations may still need provider-local scratch, so
 * this is a buffer-reuse contract rather than an allocation guarantee.
 */
public fun F64PivotedSymmetricIndefiniteDecomposition.refactorInto(
    a: F64DenseMatrix,
    workspace: Workspace? = null,
): F64PivotedSymmetricIndefiniteDecomposition = koblas.pivotedSymmetricIndefiniteInto(a, this, workspace)

/** Solve `A · x = b` for this symmetric indefinite factorization; see [F64Decompositions.solve]. */
public fun F64PivotedSymmetricIndefiniteDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** Solve into [out], which may alias [b], retaining the caller's destination buffer. */
public fun F64PivotedSymmetricIndefiniteDecomposition.solveInto(b: DoubleArray, out: DoubleArray): DoubleArray =
    koblas.solveInto(this, b, out)

/** Solve `A · X = B` for the columns of [b] at once (LAPACK `dsytrs` with `nrhs`). */
public fun F64PivotedSymmetricIndefiniteDecomposition.solve(b: F64DenseMatrix): F64DenseMatrix = koblas.solve(this, b)

/**
 * Solve every column of [b] into [out], which may be [b]. [workspace] supplies the column staging buffers
 * used by providers that do not have a block solve.
 */
public fun F64PivotedSymmetricIndefiniteDecomposition.solveInto(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solveInto(this, b, out, workspace)

/**
 * Replaces this QR factorization with that of [a], retaining this decomposition's factor buffers. [a] must
 * have this factorization's shape; implementations may still need provider-local scratch, so this is a
 * buffer-reuse contract rather than an allocation guarantee.
 */
public fun F64QrDecomposition.refactorInto(a: F64DenseMatrix, workspace: Workspace? = null): F64QrDecomposition =
    koblas.qrInto(a, this, workspace)

/** Solve this QR factorization; [minimumNorm] solves a wide original system from `qr(Aᵀ)`. */
public fun F64QrDecomposition.solve(
    b: DoubleArray,
    minimumNorm: Boolean = false,
    workspace: Workspace? = null,
): DoubleArray = koblas.solve(this, b, minimumNorm, workspace)

/**
 * Solve into [out], retaining the caller's destination. [workspace] supplies the intermediate needed to apply
 * `Q` or `Qᵀ`; its required size depends on whether [minimumNorm] is selected.
 */
public fun F64QrDecomposition.solveInto(
    b: DoubleArray,
    out: DoubleArray,
    minimumNorm: Boolean = false,
    workspace: Workspace? = null,
): DoubleArray = koblas.solveInto(this, b, out, minimumNorm, workspace)

/** Solve every right-hand-side column at once; [minimumNorm] solves a wide original system from `qr(Aᵀ)`. */
public fun F64QrDecomposition.solve(
    b: F64DenseMatrix,
    minimumNorm: Boolean = false,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solve(this, b, minimumNorm, workspace)

/** Solve every right-hand-side column into [out], retaining the caller's destination. */
public fun F64QrDecomposition.solveInto(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    minimumNorm: Boolean = false,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solveInto(this, b, out, minimumNorm, workspace)

/** `Q · y`, or `Qᵀ · y` when [transpose], without forming `Q`; see [F64Decompositions.applyQ]. */
public fun F64QrDecomposition.applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.applyQ(
    this,
    y,
    transpose,
)

/** Apply `Q`, or `Qᵀ` when [transpose], into [out], which may alias [y]. */
public fun F64QrDecomposition.applyQInto(y: DoubleArray, out: DoubleArray, transpose: Boolean = false): DoubleArray =
    koblas.applyQInto(this, y, out, transpose)

/** Least-squares solution against this rank-revealing factorization; see [F64Decompositions.solve]. */
public fun F64PivotedQrDecomposition.solve(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
    koblas.solve(this, b, workspace)

/**
 * Solve into [out], retaining the caller's destination. [workspace] supplies the least-squares staging buffer
 * for the active backend.
 */
public fun F64PivotedQrDecomposition.solveInto(
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace? = null,
): DoubleArray = koblas.solveInto(this, b, out, workspace)

/**
 * Replaces this pivoted factorization with that of [a], retaining this decomposition's factor and pivot
 * buffers and recomputing [F64PivotedQrDecomposition.rank]. [a] must have this factorization's shape;
 * implementations may still need provider-local scratch, so this is a buffer-reuse contract rather than an
 * allocation guarantee.
 */
public fun F64PivotedQrDecomposition.refactorInto(
    a: F64DenseMatrix,
    tolerance: Double = AUTOMATIC_RANK_TOLERANCE,
    workspace: Workspace? = null,
): F64PivotedQrDecomposition = koblas.qrPivotedInto(a, this, tolerance, workspace)

/** Least-squares solution for every right-hand-side column at once; see [F64Decompositions.solve]. */
public fun F64PivotedQrDecomposition.solve(b: F64DenseMatrix, workspace: Workspace? = null): F64DenseMatrix =
    koblas.solve(this, b, workspace)

/** Solve every right-hand-side column into [out], retaining the caller's destination. */
public fun F64PivotedQrDecomposition.solveInto(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solveInto(this, b, out, workspace)

/** `Q · y`, or `Qᵀ · y` when [transpose], from the nested factorization; see [F64Decompositions.applyQ]. */
public fun F64PivotedQrDecomposition.applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray =
    koblas.applyQ(this, y, transpose)

/** Apply `Q`, or `Qᵀ` when [transpose], into [out], which may alias [y]. */
public fun F64PivotedQrDecomposition.applyQInto(
    y: DoubleArray,
    out: DoubleArray,
    transpose: Boolean = false,
): DoubleArray = koblas.applyQInto(this, y, out, transpose)
