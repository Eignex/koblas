// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

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

/** Solve `A · x = b` for this factorization with the active backend; see [F64Decompositions.solve]. */
public fun F64CholeskyDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** `A⁻¹` from this factorization with the active backend; see [F64Decompositions.invert]. */
public fun F64CholeskyDecomposition.invert(workspace: Workspace? = null): F64DenseMatrix = koblas.invert(
    this,
    workspace,
)
