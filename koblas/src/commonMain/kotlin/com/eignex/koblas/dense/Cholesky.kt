// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.MatrixView
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
 * Throws at the first non-positive pivot unless [policy] says otherwise; see [CholeskyPolicy]. The old
 * default regularized silently, which returned a factor of a matrix that was not the one passed in.
 *
 * Historic note: what used to be `cholesky(regularizeNonPD = true)` is now
 * entry instead of a crash. Pass `false` for strict validation: the call then throws
 * [IllegalArgumentException] at the first non-PD pivot.
 */
fun MatrixView.cholesky(policy: CholeskyPolicy = CholeskyPolicy.Strict): DenseMatrix = koblas.cholesky(this, policy)

/** Solve `A * x = b` given `L = chol(A)`; see [Lapack.solveSpd]. */
fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray = koblas.solveSpd(L, b)

/** Invert an SPD matrix from its Cholesky factor; see [Lapack.invertSpd]. */
fun invertSpd(L: DenseMatrix, workspace: Workspace? = null): DenseMatrix = koblas.invertSpd(L, workspace)
