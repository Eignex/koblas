// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/**
 * Cholesky factorization `A = L·Lᵀ` with the active backend ([koblas]); see [Lapack.cholesky]. Only the
 * lower triangle is read, and a non-positive pivot throws [NotPositiveDefinite] unless [policy] regularizes.
 */
public fun F64DenseMatrix.cholesky(policy: CholeskyPolicy = CholeskyPolicy.Strict): CholeskyDecomposition =
    koblas.cholesky(this, policy)

/** Solve `A · x = b` for this factorization with the active backend; see [Lapack.solve]. */
public fun CholeskyDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** `A⁻¹` from this factorization with the active backend; see [Lapack.invert]. */
public fun CholeskyDecomposition.invert(workspace: Workspace? = null): F64DenseMatrix = koblas.invert(this, workspace)
