package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix

/**
 * A general LU factorization with partial pivoting, `P·A = L·U`, packed column-major with `L` below the
 * diagonal and `U` on and above. [lu] is a live buffer, not a copy, so treat it as read-only.
 *
 * @property rows the factored matrix's row count.
 * @property cols the factored matrix's column count.
 * @property lu the packed `L`\`U` factors, column-major, length `rows * cols`.
 * @param piv the live normalized 0-based row permutation, kept internally for [rowPermutation] and [rowAt].
 * @param failedAt the position of the zero pivot, or [NOT_SINGULAR]; readable afterwards through [failedAt].
 */
public class F64LuDecomposition @UnsafeKoblasApi constructor(
    public val rows: Int,
    public val cols: Int,
    @property:UnsafeKoblasApi public val lu: DoubleArray,
    piv: IntArray,
    failedAt: Int = NOT_SINGULAR,
) {
    /** The mutable factorization buffer for internal implementations. */
    internal val mutablePivots: IntArray = piv

    /** Compatibility name for the number of DGETRF pivot steps, [order]. */
    public val n: Int get() = order

    /** Number of packed diagonal entries and DGETRF pivot steps. */
    public val order: Int get() = minOf(rows, cols)

    /** Whether the source matrix was square. Square-only operations require this. */
    public val square: Boolean get() = rows == cols

    /** Number of nonzero diagonal entries in the computed factors. DGETRF is not rank-revealing. */
    public val rank: Int get() = (0 until order).count { lu[it + it * rows] != 0.0 }

    /** Whether the diagonal-pivot [rank] is below [order]. */
    public val rankDeficient: Boolean get() = rank < order

    /**
     * Legacy spelling for [rankDeficient], read straight off [failedAt] rather than rescanning the diagonal
     * so hot paths like [F64Decompositions.rcond] stay allocation-free. For a rectangular factor this means
     * a zero DGETRF pivot, not that the rectangular matrix lacks a two-sided inverse.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    /**
     * Compatibility constructor for square factors. New code should provide [rows] and [cols] explicitly.
     */
    @UnsafeKoblasApi
    public constructor(n: Int, lu: DoubleArray, piv: IntArray, failedAt: Int = NOT_SINGULAR) :
        this(n, n, lu, piv, failedAt)

    /**
     * The position k whose pivot U(k, k) was exactly zero, or [NOT_SINGULAR]. This is `dgetrf`'s positive
     * `info` made 0-based.
     */
    public var failedAt: Int = failedAt
        internal set

    init {
        requireShape(lu.size == rows * cols) { "lu length ${lu.size} != ${rows * cols}" }
        requireShape(mutablePivots.size == rows) { "piv length ${mutablePivots.size} != $rows" }
    }

    /** Extracts the `rows × order` unit-lower trapezoid `L` as an independent matrix. */
    public fun lower(): F64DenseMatrix = F64DenseMatrix(rows, order).also { lower ->
        for (j in 0 until order) {
            lower[j, j] = 1.0
            for (i in j + 1 until rows) lower[i, j] = lu[i + j * rows]
        }
    }

    /** Extracts the `order × cols` upper trapezoid `U` as an independent matrix. */
    public fun upper(): F64DenseMatrix = F64DenseMatrix(order, cols).also { upper ->
        for (j in 0 until cols) for (i in 0..minOf(j, order - 1)) upper[i, j] = lu[i + j * rows]
    }

    /** A safe extracted copy of the lower trapezoid; shorthand for [lower]. */
    public val l: F64DenseMatrix get() = lower()

    /** A safe extracted copy of the upper trapezoid; shorthand for [upper]. */
    public val u: F64DenseMatrix get() = upper()

    /** Returns a safe copy of the normalized row permutation, where entry `k` is the original row now at
     *  position `k`. */
    public fun permutation(): IntArray = mutablePivots.copyOf()

    /** A safe snapshot of the original row at every current row position; shorthand for [permutation]. */
    public val rowPermutation: IntArray get() = permutation()

    /** The original 0-based row now at factor row [position]. */
    public fun rowAt(position: Int): Int {
        requireInBounds(position, rows)
        return mutablePivots[position]
    }
}

/**
 * `det(A)` as sign(P) times the product of the U(k, k).
 *
 * The product is unscaled, so it saturates in both directions well before n is large: a 200x200 with 0.01
 * on the diagonal returns `0.0` and one with 100.0 returns infinity, neither of them singular. So a
 * returned `0.0` does not mean singular, even though a singular factorization does return it. Test
 * [F64LuDecomposition.singular] for exact singularity and [F64Decompositions.rcond] for how close to it a matrix
 * is; that is what LAPACK offers too, which ships `dgecon` and no determinant routine at all.
 */
public fun F64LuDecomposition.determinant(): Double {
    requireLuSquare(this, "determinant")
    if (singular) return 0.0
    var d = permutationSign(mutablePivots)
    for (k in 0 until order) d *= lu[k + k * rows]
    return d
}

/** Rejects a square-only LU operation before it consults a rectangular factor's pivots or diagonal. */
internal fun requireLuSquare(lu: F64LuDecomposition, routine: String) {
    requireShape(lu.square) { "$routine requires a square LU factorization; got ${lu.rows}x${lu.cols}" }
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
