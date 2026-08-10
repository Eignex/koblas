// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.NotPositiveDefinite
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
 * Cholesky factorization `A = L·Lᵀ` with the active backend ([koblas]); see [Lapack.cholesky].
 *
 * Throws [com.eignex.koblas.NotPositiveDefinite] at the first non-positive pivot. Pass
 * [CholeskyPolicy.Regularize] for a
 * factorization that floors such a pivot and continues, which is a factor of a nearby matrix rather than of
 * the one passed in — useful for an estimate that has drifted slightly indefinite, and a decision the caller
 * makes rather than a default. Catching the exception is the other way to make that decision, after the
 * fact rather than in advance.
 */
public fun DenseMatrix.cholesky(policy: CholeskyPolicy = CholeskyPolicy.Strict): CholeskyDecomposition =
    koblas.cholesky(this, policy)

/** Solve `A · x = b` for this factorization with the active backend; see [Lapack.solve]. */
public fun CholeskyDecomposition.solve(b: DoubleArray): DoubleArray = koblas.solve(this, b)

/** `A⁻¹` from this factorization with the active backend; see [Lapack.invert]. */
public fun CholeskyDecomposition.invert(workspace: Workspace? = null): DenseMatrix = koblas.invert(this, workspace)
