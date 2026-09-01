package com.eignex.koblas.dense

import com.eignex.koblas.*

/**
 * A symmetric indefinite factorization `A = L·D·Lᵀ` in LAPACK `dsytrf` lower packed form. [ldl] and [ipiv]
 * are live buffers, not copies, so treat them as read-only.
 *
 * @property n the matrix dimension.
 * @property ldl the packed factors, column-major, length `n * n`; only the lower triangle is meaningful.
 * @property ipiv LAPACK `dsytrf` 1-based pivots: positive marks a 1×1 block, a negative pair a 2×2 block.
 * @param failedAt the zero pivot position, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64PivotedSymmetricIndefiniteDecomposition @UnsafeKoblasApi constructor(
    public val n: Int,
    @property:UnsafeKoblasApi public val ldl: DoubleArray,
    @property:UnsafeKoblasApi public val ipiv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    /** The position whose pivot was exactly zero, or [NOT_SINGULAR]. This is `dsytrf`'s positive `info` made
     *  0-based. */
    public var failedAt: Int = failedAt
        internal set

    /** Whether a zero pivot was encountered. */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    init {
        requireShape(ldl.size == n * n) { "ldl length ${ldl.size} != ${n * n}" }
        requireShape(ipiv.size == n) { "ipiv length ${ipiv.size} != $n" }
    }
}
