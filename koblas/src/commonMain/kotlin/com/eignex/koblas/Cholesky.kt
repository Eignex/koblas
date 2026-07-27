// math convention: single-letter matrices L, M, etc.
@file:Suppress("VariableNaming", "FunctionParameterNaming", "PropertyName")

package com.eignex.koblas

import kotlin.math.absoluteValue
import kotlin.math.sqrt

// Cholesky helpers operating on the flat-`DoubleArray` backing of [DenseMatrix].
// Operates on the flat DoubleArray backing of [DenseMatrix].
//
// Convention: lower-triangular factor `L` with `A = L * LT`. Entry `(i, k)` for
// `k <= i` lives at `data[i * cols + k]`; entries above the diagonal are neither
// read nor written.
//
// Inner loops reduce to [denseDot] / [denseAxpy] on contiguous row runs (SIMD on
// JVM). The Givens rotation step in [choleskyDowndateInPlace] has a loop-carried
// dependency and stays scalar.

/**
 * Lower-triangular Cholesky decomposition `A = L * LT`, returned as a fresh matrix.
 *
 * With [regularizeNonPD] true (default), non-positive-definite inputs get a small
 * positive diagonal entry instead of a crash - suitable for the online stats that
 * call this on drifting precision matrices. Pass `false` for strict
 * positive-definiteness validation (e.g. user-supplied prior covariance); the
 * function throws [IllegalArgumentException] at the first non-PD pivot.
 */
fun MatrixView.cholesky(regularizeNonPD: Boolean = true): DenseMatrix {
    require(rows == cols) { "cholesky requires a square matrix; got ${rows}x$cols" }
    val n = rows
    val L = DenseMatrix(n, n)
    val Ld = L.data
    // Seed L's lower triangle with A's entries (bulk row copies for a dense source), then eliminate
    // in place: each seeded A[i,j] is consumed exactly when that slot is overwritten with L[i,j].
    // Keeps the hot loop free of per-element MatrixView dispatch.
    if (this is DenseMatrix) {
        for (i in 0 until n) data.copyInto(Ld, i * n, i * n, i * n + i + 1)
    } else {
        for (i in 0 until n) for (j in 0..i) Ld[i * n + j] = this[i, j]
    }
    for (i in 0 until n) {
        val rowI = i * n
        for (j in 0..i) {
            val rowJ = j * n
            val sum = denseDot(Ld, rowI, Ld, rowJ, j)
            if (i == j) {
                Ld[rowI + i] = sqrt(Ld[rowI + i] - sum)
            } else {
                Ld[rowI + j] = (Ld[rowI + j] - sum) / Ld[rowJ + j]
            }
        }
        if (Ld[rowI + i] <= 0.0 || Ld[rowI + i].isNaN()) {
            require(regularizeNonPD) {
                "matrix is not positive-definite at pivot $i (diagonal=${Ld[rowI + i]})"
            }
            Ld[rowI + i] = 1e-5
        }
    }
    return L
}

/**
 * In-place Cholesky downdate of a lower-triangular factor: modifies [this] so that
 * the matrix `A = L * LT` it represents becomes `A - x * xT`. Returns `0.0` on
 * success, or a positive "norm" value when the downdate would leave the matrix
 * outside the positive-definite cone. The caller then has to repair via a fresh
 * decomposition or take a smaller step.
 *
 * Algorithm: solve `L * s = x` by forward substitution; if `||s|| < 1` the downdate
 * stays SPD. Then apply Givens rotations to the rows of L (the natural direction
 * for lower-triangular storage) to absorb `s` without breaking triangularity.
 */
fun DenseMatrix.choleskyDowndateInPlace(x: VectorView): Double {
    require(rows == cols) { "choleskyDowndateInPlace requires a square matrix; got ${rows}x$cols" }
    require(rows == x.size) { "x size ${x.size} must match matrix dim $rows" }
    if (rows == 0) return 0.0 // an empty downdate stays in the cone trivially
    val L = data
    val n = rows
    val s = DoubleArray(n)
    val c = DoubleArray(n)

    // Solve L * s = x by forward substitution. Inner sum is a contiguous dot product.
    s[0] = x[0] / L[0]
    for (i in 1 until n) {
        val rowI = i * n
        val sum = denseDot(L, rowI, s, 0, i)
        s[i] = (x[i] - sum) / L[rowI + i]
    }

    val norm = norm2(DenseVector.wrap(s))
    if (norm <= 0.0 || norm >= 1.0) return norm

    var alpha = sqrt(1.0 - norm * norm)
    for (ii in 0 until n) {
        val i = n - ii - 1
        val scale = alpha + s[i].absoluteValue
        val a = alpha / scale
        val b = s[i] / scale
        val nrm = sqrt(a * a + b * b)
        c[i] = a / nrm
        s[i] = b / nrm
        alpha = scale * nrm
    }
    // Apply rotations along rows of L. Loop-carried in xx - stays scalar.
    for (j in 0 until n) {
        val rowJ = j * n
        var xx = 0.0
        for (ii in 0..j) {
            val i = j - ii
            val idx = rowJ + i
            val t = c[i] * xx + s[i] * L[idx]
            L[idx] = c[i] * L[idx] - s[i] * xx
            xx = t
        }
    }
    return 0.0
}

/**
 * Solve `A * x = b` for `x`, given `L = chol(A)` (lower-triangular, `A = L * LT`).
 * Allocates a fresh result vector; [b] is not modified.
 *
 * Forward substitution `L * y = b` runs over contiguous row data and uses [denseDot].
 * Back substitution `LT * x = y` is column-oriented: once `x[i]` is final, its contribution
 * is subtracted from the remaining right-hand side along contiguous row `i` via [denseAxpy].
 */
fun solveSpd(L: DenseMatrix, b: DoubleArray): DoubleArray {
    val n = L.rows
    require(b.size == n) { "solveSpd: b size ${b.size}, expected $n" }
    val x = b.copyOf()
    trsvCore(L.data, n, x, lower = true, transpose = false, unitDiag = false)
    trsvCore(L.data, n, x, lower = true, transpose = true, unitDiag = false)
    return x
}

/**
 * Invert an SPD matrix from its Cholesky factor: returns `A^-1` given `L = chol(A)`.
 *
 * Solves `A * x = e_j` column by column, exploiting the unit-vector right-hand side: forward
 * substitution starts at row `j` (the leading entries are provably zero) and back substitution
 * only produces rows `>= j` — the strictly-upper entries of the symmetric `A^-1` come from
 * mirroring the lower triangle.
 */
fun invertSpd(L: DenseMatrix): DenseMatrix {
    val n = L.rows
    val Ld = L.data
    val inv = DenseMatrix(n, n)
    val invd = inv.data
    val y = DoubleArray(n)
    for (j in 0 until n) {
        // L y = e_j: rows before j stay zero.
        y[j] = 1.0 / Ld[j * n + j]
        for (i in j + 1 until n) {
            val rowI = i * n
            y[i] = -denseDot(Ld, rowI + j, y, j, i - j) / Ld[rowI + i]
        }
        // LT x = y, column-oriented, restricted to rows >= j.
        for (i in n - 1 downTo j) {
            val xi = y[i] / Ld[i * n + i]
            y[i] = xi
            if (xi != 0.0) denseAxpy(y, j, -xi, Ld, i * n + j, i - j)
        }
        for (i in j until n) {
            invd[i * n + j] = y[i]
            invd[j * n + i] = y[i]
        }
    }
    return inv
}
