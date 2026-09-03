// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming")

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix

/**
 * Cholesky factorization `A = L·Lᵀ` with the active backend ([koblas]). [lower] selects the authoritative
 * triangle without checking the other.
 * A non-positive pivot throws [NotPositiveDefinite] unless [policy] regularizes.
 */
public fun F64DenseMatrix.cholesky(
    policy: CholeskyPolicy = CholeskyPolicy.Strict,
    lower: Boolean = true,
): F64CholeskyDecomposition = koblas.cholesky(asLowerSymmetricInput(lower, "cholesky"), policy)

/**
 * Replaces this Cholesky factorization with that of [a], retaining this decomposition's factor buffer. [a]'s
 * dimension must match this factorization's; implementations may still need provider-local scratch, so this
 * is a buffer-reuse contract rather than an allocation guarantee.
 * @throws NotPositiveDefinite at the first non-positive pivot unless [policy] regularizes.
 */
public fun F64CholeskyDecomposition.refactorInto(
    a: F64DenseMatrix,
    policy: CholeskyPolicy = CholeskyPolicy.Strict,
): F64CholeskyDecomposition = koblas.choleskyInto(a, this, policy)

/** Solve `A · x = b` for this factorization with the active backend; see [F64Decompositions.solve]. */
public fun F64CholeskyDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** Solve into [out], which may alias [b], retaining the caller's destination buffer. */
public fun F64CholeskyDecomposition.solveInto(b: DoubleArray, out: DoubleArray): DoubleArray = koblas.solveInto(
    this,
    b,
    out,
)

/** Solve `A · X = B` for this factorization with the active backend; see [F64Decompositions.solve]. */
public fun F64CholeskyDecomposition.solve(b: F64DenseMatrix): F64DenseMatrix = koblas.solve(this, b)

/**
 * Solve every column of [b] into [out], which may be [b]. [workspace] provides staging for providers that
 * solve one right-hand side at a time.
 */
public fun F64CholeskyDecomposition.solveInto(
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    workspace: Workspace? = null,
): F64DenseMatrix = koblas.solveInto(this, b, out, workspace)

/**
 * Update this factorization in place so it factors `A + sigma * v * vT`, returning it; see
 * [F64Decompositions.choleskyRank1Update].
 */
public fun F64CholeskyDecomposition.rank1Update(
    v: DoubleArray,
    sigma: Double = 1.0,
    workspace: Workspace? = null,
): F64CholeskyDecomposition = koblas.choleskyRank1Update(this, v, sigma, workspace)

/** `A⁻¹` from this factorization with the active backend; see [F64Decompositions.invert]. */
public fun F64CholeskyDecomposition.invert(workspace: Workspace? = null): F64DenseMatrix = koblas.invert(
    this,
    workspace,
)

/** `A⁻¹` into [out], which is returned, without allocating a result; see [F64Decompositions.invertInto]. */
public fun F64CholeskyDecomposition.invertInto(out: F64DenseMatrix, workspace: Workspace? = null): F64DenseMatrix =
    koblas.invertInto(this, out, workspace)
