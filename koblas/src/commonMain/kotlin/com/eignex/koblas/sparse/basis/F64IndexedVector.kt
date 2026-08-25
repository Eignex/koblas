package com.eignex.koblas.sparse.basis

import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireShape
import kotlin.math.abs

/**
 * A length-[size] vector that carries the positions of its own nonzeros, the form a basis solve reads and
 * writes.
 *
 * The values are held densely and [indices] names where the nonzeros are, so a solve whose result has a
 * handful of entries is produced and consumed without a pass over all [size] positions. The invariant runs
 * one way: every nonzero appears among the first [count] entries of [indices], but an index outlives its
 * value cancelling to zero, because a triangular sweep pays more to notice the cancellation than the stale
 * index costs. [tighten] restores the exact set.
 *
 * This is mutable scratch. A solve overwrites it in place and a simplex reuses one across iterations rather
 * than allocating per solve, so one vector is driven by one thread.
 *
 * @property size the logical length, counting the unstored zeros.
 */
@OptIn(UnsafeKoblasApi::class)
public class F64IndexedVector(public val size: Int) {
    init {
        requireShape(size >= 0) { "negative size: $size" }
    }

    /** Live dense storage, length [size], zero at every position not stored; do not mutate. */
    @UnsafeKoblasApi
    public val values: DoubleArray = DoubleArray(size)

    /** Live stored positions, the first [count] entries of a length-[size] array; do not mutate. */
    @UnsafeKoblasApi
    public val indices: IntArray = IntArray(size)

    /** How many positions [indices] names. */
    public var count: Int = 0
        private set

    /** The fraction of [size] that is stored, what a solve reads to choose between its sweeps. */
    public val density: Double get() = if (size == 0) 0.0 else count.toDouble() / size

    /** The value at [i], zero where nothing is stored. */
    public operator fun get(i: Int): Double {
        requireInBounds(i, size)
        return values[i]
    }

    /**
     * Visits the stored entries as `(index, value)`, in the order they were stored rather than ascending.
     * A position whose value has cancelled to zero is visited like any other.
     */
    public inline fun forEachStored(block: (i: Int, v: Double) -> Unit) {
        val stored = indices
        val entries = values
        for (k in 0 until count) block(stored[k], entries[stored[k]])
    }

    /** Zeros every stored position and empties the index set. Costs [count], not [size]. */
    public fun clear() {
        for (k in 0 until count) values[indices[k]] = 0.0
        count = 0
    }

    /**
     * Writes [v] at [i] and records the position, which must not already be stored.
     *
     * Recording rather than searching is what makes scattering a column `O(nnz)`, so a caller assembling a
     * vector from a source with repeated positions sums them first.
     */
    public fun store(i: Int, v: Double) {
        requireInBounds(i, size)
        values[i] = v
        indices[count] = i
        count++
    }

    /** Empties this vector and stores the unit vector `e(i)`. */
    public fun unit(i: Int) {
        clear()
        store(i, 1.0)
    }

    /** Empties this vector and stores the nonzeros of column [j] of [a], whose rows must be [size]. */
    public fun scatterColumn(a: F64SparseMatrix, j: Int) {
        requireShape(a.rows == size) { "scatterColumn: rows ${a.rows} != $size" }
        requireInBounds(j, a.cols)
        clear()
        a.forEachInColumn(j) { i, v -> if (v != 0.0) store(i, v) }
    }

    /** Empties this vector and stores the nonzeros of [dense], which must have length [size]. */
    public fun scatter(dense: DoubleArray) {
        requireShape(dense.size == size) { "scatter: dense size ${dense.size} != $size" }
        clear()
        for (i in 0 until size) if (dense[i] != 0.0) store(i, dense[i])
    }

    /** Writes this vector densely into [out], which must have length [size] and is returned. */
    public fun gather(out: DoubleArray): DoubleArray {
        requireShape(out.size == size) { "gather: out size ${out.size} != $size" }
        out.fill(0.0)
        for (k in 0 until count) out[indices[k]] = values[indices[k]]
        return out
    }

    /** This vector densely in a fresh array. */
    public fun toDoubleArray(): DoubleArray = gather(DoubleArray(size))

    /**
     * Drops stored positions whose magnitude is at or below [tolerance], so the index set holds only
     * entries still carrying weight.
     */
    public fun tighten(tolerance: Double = 0.0) {
        requireShape(tolerance >= 0.0) { "negative tolerance: $tolerance" }
        var kept = 0
        for (k in 0 until count) {
            val i = indices[k]
            if (abs(values[i]) > tolerance) {
                indices[kept] = i
                kept++
            } else {
                values[i] = 0.0
            }
        }
        count = kept
    }

    /**
     * Rebuilds the index set by scanning all [size] positions, for a caller that wrote [values] past the
     * seam and left the set behind.
     */
    public fun reindex() {
        count = 0
        for (i in 0 until size) if (values[i] != 0.0) store(i, values[i])
    }

    override fun toString(): String = "F64IndexedVector(size=$size, count=$count)"
}
