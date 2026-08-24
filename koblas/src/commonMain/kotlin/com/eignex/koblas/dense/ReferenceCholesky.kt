@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, L

package com.eignex.koblas.dense

import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.requireShape
import kotlin.math.sqrt

/*
 * The portable Cholesky factorization and the SPD inverse over its factor, netlib dpotrf and dpotri. This is
 * the semantic definition a native Cholesky is validated against; [F64ReferenceLapack] is the F64Lapack
 * surface over it.
 */

/** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
 *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
 *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
 */
internal fun referenceCholesky(
    kernels: F64VectorKernels,
    a: F64DenseMatrix,
    policy: CholeskyPolicy,
): F64CholeskyDecomposition {
    requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
    val n = a.rows
    val l = F64DenseMatrix(n, n)
    val ld = l.data
    for (j in 0 until n) a.data.copyInto(ld, j + j * n, j + j * n, (j + 1) * n)
    // Left-looking column Cholesky: column j gathers what every column before it owes, rather than
    // each column pushing its contribution out to the ones after it.
    for (j in 0 until n) {
        val base = j + j * n
        val len = n - j
        for (p in 0 until j) {
            val f = ld[j + p * n]
            if (f != 0.0) kernels.axpy(ld, base, -f, ld, j + p * n, len)
        }
        val pivot = ld[base]
        if (pivot <= 0.0 || pivot.isNaN()) {
            if (policy !is CholeskyPolicy.Regularize) {
                throw NotPositiveDefinite(
                    j,
                    pivot,
                    "matrix is not positive-definite at pivot $j (diagonal=$pivot); pass " +
                        "CholeskyPolicy.Regularize to factor a nearby matrix instead",
                )
            }
            ld[base] = sqrt(policy.minimumPivot)
        } else {
            ld[base] = sqrt(pivot)
        }
        val diag = ld[base]
        for (i in base + 1 until base + len) ld[i] = ld[i] / diag
    }
    return F64CholeskyDecomposition(l)
}

/** Invert an SPD matrix from its Cholesky factorization, returning `A⁻¹` given [chol] (LAPACK `dpotri`). */
internal fun referenceSpdInvert(
    kernels: F64VectorKernels,
    chol: F64CholeskyDecomposition,
    workspace: Workspace?,
): F64DenseMatrix {
    val n = chol.n
    val ld = chol.l.data
    val inv = F64DenseMatrix(n, n)
    val invd = inv.data
    val y = workspace?.take(n) ?: DoubleArray(n)
    for (j in 0 until n) {
        y.fill(0.0, j, n)
        y[j] = 1.0
        for (c in j until n) {
            val base = c + c * n
            val yc = y[c] / ld[base]
            y[c] = yc
            if (yc != 0.0) kernels.axpy(y, c + 1, -yc, ld, base + 1, n - c - 1)
        }
        for (i in n - 1 downTo j) {
            val base = i + i * n
            y[i] = (y[i] - kernels.dot(ld, base + 1, y, i + 1, n - i - 1)) / ld[base]
        }
        for (i in j until n) {
            invd[i + j * n] = y[i]
            invd[j + i * n] = y[i]
        }
    }
    workspace?.release(y)
    return inv
}
