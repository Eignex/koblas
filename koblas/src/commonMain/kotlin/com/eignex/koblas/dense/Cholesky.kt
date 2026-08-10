// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.MatrixLike
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

// Cholesky helpers operating on the flat-`DoubleArray` backing of [DenseMatrix].
// Operates on the flat DoubleArray backing of [DenseMatrix].
//
// Convention: lower-triangular factor `L` with `A = L * LT`. Entry `(i, k)` for
// `k <= i` lives at `data[i + k * rows]`; entries above the diagonal are neither
// read nor written. Column `k` of the factor is therefore the contiguous run
// `data[k + k * rows until (k + 1) * rows]`.
//
// Inner loops reduce to [VectorKernels.dot] / [VectorKernels.axpy] on contiguous column runs (SIMD
// on JVM).

/**
 * Lower-triangular Cholesky decomposition `A = L * LT`, returned as a fresh matrix, from the installed
 * backend; see [Lapack.cholesky].
 *
 * Throws [com.eignex.koblas.NotPositiveDefinite] at the first non-positive pivot. Pass
 * [CholeskyPolicy.Regularize] for a
 * factorization that floors such a pivot and continues, which is a factor of a nearby matrix rather than of
 * the one passed in — useful for an estimate that has drifted slightly indefinite, and a decision the caller
 * makes rather than a default.
 */
fun MatrixLike.cholesky(policy: CholeskyPolicy = CholeskyPolicy.Strict): DenseMatrix = koblas.cholesky(this, policy)

/** Solve `A * x = b` given `L = chol(A)`; see [Lapack.solveSpd]. */
fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray = koblas.solveSpd(L, b)

/** Invert an SPD matrix from its Cholesky factor; see [Lapack.invertSpd]. */
fun invertSpd(L: DenseMatrix, workspace: Workspace? = null): DenseMatrix = koblas.invertSpd(L, workspace)

/** `A⁻¹` from an LU factorization (LAPACK `dgetri`); see [LinearAlgebra.invert]. */
fun invert(lu: LuDecomposition, workspace: Workspace? = null): DenseMatrix = koblas.invert(lu, workspace)
