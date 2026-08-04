package com.eignex.koblas.dense

import com.eignex.koblas.NOT_SINGULAR

/**
 * A symmetric indefinite factorization `A = L·D·Lᵀ` with Bunch–Kaufman partial pivoting in LAPACK
 * `dsytrf` (lower) packed form: [ldl] is the `n×n` column-major buffer whose lower triangle holds the
 * unit-lower `L` columns and the 1×1/2×2 diagonal blocks of `D` (the strictly upper triangle is
 * untouched input), and [ipiv] uses the LAPACK convention — `ipiv[k] > 0` marks a 1×1 block with row
 * interchange `k ↔ ipiv[k]−1`, while `ipiv[k] == ipiv[k+1] < 0` marks a 2×2 block at `(k, k+1)` with
 * interchange `k+1 ↔ −ipiv[k]−1`. Produced by [LinearAlgebra.ldl]; consumed by [LinearAlgebra.solve].
 *
 * This is the KKT-system kernel interior-point and QP methods build on: it factors symmetric matrices
 * that are indefinite, where [cholesky] does not apply. As with [LuDecomposition], the
 * format is shared across backends, so a factorization from one backend solves correctly on another.
 * [ldl] and [ipiv] are live buffers, not copies — treat them as read-only.
 *
 * @property n the matrix dimension.
 * @property ldl the packed factors, column-major, length `n * n`; only the lower triangle is meaningful.
 * @property ipiv the pivot record, LAPACK `dsytrf` convention, 1-based entries.
 * @property failedAt the position of the zero pivot, or [NOT_SINGULAR] — `dsytrf`'s positive `info` made
 *   0-based, and the same convention [LuDecomposition] and the sparse factorizations report.
 */
class LdlDecomposition(val n: Int, val ldl: DoubleArray, val ipiv: IntArray, val failedAt: Int = NOT_SINGULAR) {
    /** Whether a zero pivot was encountered; solving a singular factorization is not meaningful. Derived
     *  from [failedAt], for the reason [LuDecomposition.singular] gives. */
    val singular: Boolean get() = failedAt != NOT_SINGULAR

    init {
        require(ldl.size == n * n) { "ldl length ${ldl.size} != ${n * n}" }
        require(ipiv.size == n) { "ipiv length ${ipiv.size} != $n" }
    }
}
