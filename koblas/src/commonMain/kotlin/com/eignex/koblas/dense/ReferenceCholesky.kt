@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, L

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.sqrt

/*
 * The portable Cholesky factorization and the SPD inverse over its factor, netlib dpotrf and dpotri. This is
 * the semantic definition a native Cholesky is validated against; [F64ReferenceDecompositions] is the F64Decompositions
 * surface over it.
 */

/** Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite [a] (`dpotrf` with `uplo = 'L'`).
 *  Reads only the lower triangle of [a], so an upper-only matrix factors to silent nonsense.
 *  @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
 */
internal fun referenceCholesky(
    kernels: F64Kernels,
    a: F64DenseMatrix,
    policy: CholeskyPolicy,
): F64CholeskyDecomposition = referenceCholeskyInto(
    kernels,
    a,
    F64CholeskyDecomposition(F64DenseMatrix(a.rows, a.rows)),
    policy,
)

/**
 * [referenceCholesky] into [out]'s existing factor buffer instead of a fresh one, discarding what [out] held.
 * [out] may be backed by [a]'s own buffer, which makes the gather below a no-op over the triangle it reads.
 *
 * @throws com.eignex.koblas.NotPositiveDefinite at the first non-positive pivot unless [policy] allows it.
 */
internal fun referenceCholeskyInto(
    kernels: F64Kernels,
    a: F64DenseMatrix,
    out: F64CholeskyDecomposition,
    policy: CholeskyPolicy,
): F64CholeskyDecomposition {
    requireShape(a.rows == a.cols) { "cholesky requires a square matrix; got ${a.rows}x${a.cols}" }
    requireShape(out.n == a.rows) { "choleskyInto: out is ${out.n}x${out.n}, expected ${a.rows}x${a.rows}" }
    val n = a.rows
    val ld = out.l.data
    // A reused destination arrives holding the previous factor where a fresh one arrives zeroed, so the
    // strict upper triangle is cleared rather than inherited.
    for (j in 0 until n) {
        ld.fill(0.0, j * n, j * n + j)
        a.data.copyInto(ld, j + j * n, j + j * n, (j + 1) * n)
    }
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
    return out
}

/** Invert an SPD matrix from its Cholesky factorization, returning `A⁻¹` given [chol] (LAPACK `dpotri`). */
internal fun referenceSpdInvert(
    kernels: F64Kernels,
    chol: F64CholeskyDecomposition,
    workspace: Workspace?,
): F64DenseMatrix {
    val n = chol.n
    val ld = chol.l.data
    val inv = F64DenseMatrix(n, n)
    val invd = inv.data
    workspace.borrow(n) { y ->
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
    }
    return inv
}
