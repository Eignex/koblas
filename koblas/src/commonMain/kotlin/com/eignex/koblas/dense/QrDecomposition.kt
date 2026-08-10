package com.eignex.koblas.dense

import com.eignex.koblas.requireShape

/**
 * A QR factorization `A = Q·R` in LAPACK `dgeqrf` packed form: [qr] is the `m×n` column-major buffer with
 * `R` on and above the diagonal and the Householder vectors below it (each vector's implicit leading 1
 * is not stored), and [tau] holds the `min(m, n)` reflector coefficients of `H_k = I − tau_k·v_k·v_kᵀ`
 * with `Q = H_0·H_1···H_{k−1}`. Produced by [LinearAlgebra.qr]; consumed by [LinearAlgebra.applyQ] and
 * [LinearAlgebra.solveLeastSquares].
 *
 * As with [LuDecomposition], the format is shared across backends — a factorization produced by one
 * backend applies and solves correctly on another. [qr] and [tau] are live buffers, not copies — treat
 * them as read-only.
 *
 * @property m the row count of the factored matrix.
 * @property n the column count of the factored matrix.
 * @property qr the packed `R` and Householder vectors, column-major, length `m * n`.
 * @property tau the reflector coefficients, length `min(m, n)`.
 */
public class QrDecomposition(
    public val m: Int,
    public val n: Int,
    public val qr: DoubleArray,
    public val tau: DoubleArray,
) {
    init {
        requireShape(qr.size == m * n) { "qr length ${qr.size} != ${m * n}" }
        requireShape(tau.size == minOf(m, n)) { "tau length ${tau.size} != ${minOf(m, n)}" }
    }
}
