package com.eignex.koblas.sparse.basis

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.requireHfactorShape
import kotlin.math.abs
import kotlin.math.max

/** What a basis solver made of an update, and what the caller owes it before the next solve. */
public enum class BasisUpdate {
    /** The factors now represent the new basis and are fit to solve against. */
    APPLIED,

    /**
     * The factors represent the new basis, but the solver would rather be rebuilt than take another
     * update. Advisory: solving against them stays correct, and the caller paces its own refactorizations.
     */
    REFACTORIZE,

    /**
     * The pivot was not usable and the factors are unchanged, so the basis and the solver have parted
     * company. Only a new refactorization recovers.
     */
    SINGULAR,
}

/**
 * Why a solver advised [BasisUpdate.REFACTORIZE], for a caller pacing its rebuilds.
 *
 * The advisory is one answer reached two ways, and which one it was is what tells a worn factorization from
 * a numerically unhappy one. A caller logging these can read the difference off a histogram rather than
 * guessing at it.
 */
public enum class RefactorizeReason {
    /**
     * The factorization itself asked, having found the update it just took unfit to build on. This is the
     * numerical cause: the factors have lost accuracy rather than merely grown.
     */
    FACTOR_ASKED,

    /**
     * The updates have cost what the factorization did, so rebuilding is now the cheaper path. This is the
     * economic cause and says nothing about accuracy.
     */
    UPDATES_WORN,
}

/**
 * How much of a basis survived triangularization into the elimination kernel.
 *
 * A factorization peels off the rows and columns it can order triangularly before choosing pivots for what
 * is left. What is left is the kernel, and its size against the basis dimension says whether a caller's
 * bases are staying mostly triangular, which a fill count alone does not.
 *
 * @property dimension the kernel's order, at most the basis dimension.
 * @property entries the stored entries in it.
 */
public class BasisKernel(public val dimension: Int, public val entries: Int)

/**
 * The basis a solver settled on, where it was allowed to repair one it could not invert as given.
 *
 * A rank-deficient basis has slots whose columns are linearly dependent. A provider that repairs one fills
 * those slots with unit columns, which are not columns of `A` at all: they are the slacks a simplex would
 * hold there. So a repaired basis is reported slot by slot rather than as column indices, since some slots
 * name nothing in the caller's matrix.
 *
 * A caller adopting one takes both arrays as its new basis. Declining it means factorizing something else;
 * the solver holds the repaired basis either way until the next rebuild.
 *
 * @property columns the column of `A` in each slot, or -1 where the slot holds a unit column.
 * @property unitRows the row each unit column stands for, or -1 where the slot holds a column of `A`.
 */
public class BasisRepair(public val columns: IntArray, public val unitRows: IntArray) {
    /** Whether anything was replaced. False means the basis factorized exactly as it was given. */
    public val repaired: Boolean get() = unitRows.any { it >= 0 }

    /** How many slots were replaced. */
    public val replacedSlots: Int get() = unitRows.count { it >= 0 }
}

/**
 * A factorization a solver set aside, to be put back later.
 *
 * Held by the solver it came from and only meaningful there. Closing one releases what it holds; closing
 * the solver releases any it still owns, so a caller that drops a search node need not unwind its snapshots
 * by hand.
 */
public interface BasisSnapshot : AutoCloseable

/** Numerically scaled residual information for one basis solve. */
public data class BasisSolveQuality(
    /** `max |B·x - b|`, or `max |Bᵀ·x - b|` for a transposed solve. */
    public val residualInfinityNorm: Double,
    /** The residual divided by `max(1, max(|B·x|, |b|))`. */
    public val relativeResidual: Double,
)

@Suppress("LongParameterList") // the basis in two arrays, its operands, and the direction
internal fun basisSolveQuality(
    a: SparseMatrix,
    basicIndex: IntArray,
    unitRows: IntArray?,
    rhs: DoubleArray,
    solution: IndexedVector,
    transpose: Boolean,
): BasisSolveQuality {
    val n = a.rows
    requireHfactorShape(basicIndex.size == n) { "basis has ${basicIndex.size} columns, expected $n" }
    requireHfactorShape(rhs.size == n) { "rhs size ${rhs.size} != $n" }
    requireHfactorShape(solution.size == n) { "solution size ${solution.size} != $n" }
    val x = solution.toDoubleArray()
    val product = DoubleArray(n)
    if (transpose) {
        for (slot in 0 until n) {
            val unit = unitRows?.get(slot) ?: -1
            if (unit >= 0) {
                product[slot] = x[unit]
                continue
            }
            var sum = 0.0
            a.forEachInColumn(basicIndex[slot]) { row, value -> sum += value * x[row] }
            product[slot] = sum
        }
    } else {
        for (slot in 0 until n) {
            val multiplier = x[slot]
            if (multiplier == 0.0) continue
            val unit = unitRows?.get(slot) ?: -1
            if (unit >= 0) {
                product[unit] += multiplier
                continue
            }
            a.forEachInColumn(basicIndex[slot]) { row, value ->
                product[row] += value * multiplier
            }
        }
    }
    var residual = 0.0
    var scale = 0.0
    for (i in 0 until n) {
        residual = max(residual, abs(product[i] - rhs[i]))
        scale = max(scale, max(abs(product[i]), abs(rhs[i])))
    }
    return BasisSolveQuality(residual, residual / max(1.0, scale))
}
