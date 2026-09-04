@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, L

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.hypot
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
    requireSquare(a, "cholesky")
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
    out: F64DenseMatrix,
    workspace: Workspace?,
): F64DenseMatrix {
    val n = chol.n
    val ld = chol.l.data
    val invd = out.data
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
    return out
}

/**
 * Rank-k update of a Cholesky factorization in place, `A + sigma·V·Vᵀ = L̃·L̃ᵀ`, returning [chol] with its
 * factor buffer rewritten. [v] holds the update vectors as [columns] consecutive runs of `n` and is not
 * modified.
 *
 * One plane rotation per row: at step k the rotation that zeroes the working vector's entry k against the
 * diagonal is applied to the column tail below it. Every rotation is orthogonal, so this needs no
 * positive-definiteness check, since `A + sigma·V·Vᵀ` is positive-definite whenever `A` is and [sigma] is
 * not negative. A rotation rather than the `(l + s·x)/c` recurrence, which divides by the diagonal entry and
 * so cannot answer for a factor carrying a zero there. The rotation is [rotg]'s, computed in the loop rather
 * than called, for the reason given there.
 */
@Suppress("LongParameterList") // the factor, the update block and its width, the scale, and scratch
internal fun referenceCholeskyRankUpdate(
    kernels: F64Kernels,
    chol: F64CholeskyDecomposition,
    v: DoubleArray,
    columns: Int,
    sigma: Double,
    workspace: Workspace?,
): F64CholeskyDecomposition {
    val n = chol.n
    if (n == 0 || columns == 0 || sigma == 0.0) return chol
    val ld = chol.l.data
    val scale = sqrt(sigma)
    workspace.borrow(n) { x ->
        for (column in 0 until columns) {
            v.copyInto(x, 0, column * n, (column + 1) * n)
            if (scale != 1.0) kernels.scale(x, 0, scale, n)
            for (k in 0 until n) {
                val base = k + k * n
                val diagonal = ld[base]
                // The rotation is computed here rather than through the public rotg, which returns an
                // F64Givens and would allocate one per column, n of them for a rank-1 update and n·k for a
                // rank-k one. Taking r as a magnitude also makes the positive diagonal a property of the
                // construction: netlib's convention takes r's sign from whichever input dominates, so a
                // larger negative working entry would put a negative number on the diagonal, and this r is
                // that one's absolute value. The quotients below are therefore already the sign-corrected
                // pair, which is why nothing is negated afterwards. hypot keeps rotg's behaviour for
                // magnitudes whose squares would overflow or vanish.
                //
                // rotg's own scaled form gives the same two guarantees, and measuring it here made this
                // slower rather than faster: the two divisions it needs cost more than the correctly
                // rounded call they replace. CholeskyUpdateBenchmark.rankUpdate at n = 256 on the reference
                // backend read 15.0 and 15.1 us/op with hypot against 17.5 and 16.9 scaled, over two runs
                // of each with the benchmark pinned to two cores; rankUpdateBlock could not separate them.
                val entry = x[k]
                val r = hypot(diagonal, entry)
                if (r != 0.0) {
                    ld[base] = r
                    val len = n - k - 1
                    // Two divisions rather than a reciprocal and two multiplies: this rotation sets the
                    // factor's numerical quality, and an extra rounding per entry compounds across n of them.
                    if (len > 0) kernels.rot(ld, base + 1, x, k + 1, len, diagonal / r, entry / r)
                }
            }
        }
    }
    return chol
}
