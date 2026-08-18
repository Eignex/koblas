package com.eignex.koblas.dense

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseLu

/**
 * A general LU factorization with partial pivoting, `P·A = L·U`, packed column-major with `L` below the
 * diagonal and `U` on and above. [lu] and [piv] are live buffers, not copies, so treat them as read-only.
 *
 * @property n the matrix dimension.
 * @property lu the packed `L`\`U` factors, column-major, length `n * n`.
 * @property piv the row permutation, where piv(k) is the original row now at position k.
 * @param failedAt the position of the zero pivot, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64LuDecomposition(
    public val n: Int,
    public val lu: DoubleArray,
    public val piv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    /**
     * The position k whose pivot U(k, k) was exactly zero, or [NOT_SINGULAR]. This is `dgetrf`'s positive
     * `info` made 0-based.
     */
    public var failedAt: Int = failedAt
        internal set

    /**
     * Whether a zero pivot was encountered. [F64LinearAlgebra.solve] then throws
     * [com.eignex.koblas.SingularMatrix] rather than dividing by it.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    init {
        requireShape(lu.size == n * n) { "lu length ${lu.size} != ${n * n}" }
        requireShape(piv.size == n) { "piv length ${piv.size} != $n" }
    }
}

/** `det(A)` as sign(P) times the product of the U(k, k), or exactly `0.0` when [F64LuDecomposition.singular].
 *  The floating-point counterpart of [F64SparseLu.determinant]. */
public fun F64LuDecomposition.determinant(): Double {
    if (singular) return 0.0
    var d = permutationSign(piv)
    for (k in 0 until n) d *= lu[k * n + k]
    return d
}

/** Sign of the permutation [p], where p(k) is the original index at position k. */
internal fun permutationSign(p: IntArray): Double {
    val seen = BooleanArray(p.size)
    var cycles = 0
    for (s in p.indices) {
        if (seen[s]) continue
        cycles++
        var i = s
        while (!seen[i]) {
            seen[i] = true
            i = p[i]
        }
    }
    return if ((p.size - cycles) % 2 == 0) 1.0 else -1.0
}
