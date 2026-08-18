package com.eignex.koblas.dense

import com.eignex.koblas.requireShape

/**
 * A QR factorization `A = Q·R` in LAPACK `dgeqrf` packed form, `R` on and above the diagonal and the
 * Householder vectors below. [qr] and [tau] are live buffers, not copies, so treat them as read-only.
 *
 * @property m the row count of the factored matrix.
 * @property n the column count of the factored matrix.
 * @property qr the packed `R` and Householder vectors, column-major, length `m * n`.
 * @property tau the coefficients of `H_k = I − tau_k·v_k·v_kᵀ`, length `min(m, n)`.
 */
public class F64QrDecomposition(
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
