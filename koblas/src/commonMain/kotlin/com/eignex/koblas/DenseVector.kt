package com.eignex.koblas

import kotlin.jvm.JvmStatic

/**
 * A vector that stores every one of its entries, addressed as a buffer, an origin and a step.
 *
 * Dense is about what is stored, not about how it is spaced: a run of adjacent entries and every second entry
 * of a longer buffer are both dense, and a BLAS call takes either as a pointer and an increment without
 * knowing the difference. That is why this is the operand type the dense matrix routines declare. A vector
 * storing a pattern is a different thing with no increment to hand over, so [SparseVector] is deliberately not
 * one of these and does not compile where one is wanted.
 *
 * Sealed because the two shapes are the whole set: entries laid out one after another, or entries a fixed
 * step apart. A caller that must tell them apart can do so exhaustively.
 *
 * Neither shape implies ownership. [of] and [zero] allocate, while [wrap] and the [StridedVector] constructor
 * borrow an array the caller keeps; mutations stay visible through every reference either way. So a vector
 * here says how its entries are spaced, and says nothing about who owns them.
 */
public sealed interface DenseVector : Vector {
    /** Borrowed or owned backing array. */
    public val values: DoubleArray

    /** First physical entry of this vector within [values]. */
    public val offset: Int

    /** Physical distance between adjacent entries, which may be negative but is never zero. */
    public val stride: Int

    /** Writes (v) at index (i). */
    public operator fun set(i: Int, v: Double)

    /**
     * Factories for dense vectors.
     *
     * Each names [ContiguousVector] rather than this interface, because every one of them produces that
     * shape and a caller that wants the serializable storage should not have to narrow the result back.
     */
    public companion object {
        /** Copy a `DoubleArray` into a fresh dense vector. */
        @JvmStatic
        public fun of(values: DoubleArray): ContiguousVector = ContiguousVector(values.copyOf())

        /** A dense vector of [size] zeros. */
        @JvmStatic
        public fun zero(size: Int): ContiguousVector {
            requireShape(size >= 0) { "negative size: $size" }
            return ContiguousVector(DoubleArray(size))
        }

        /** Wrap an existing `DoubleArray` without copying; mutations remain visible through both references. */
        @JvmStatic
        public fun wrap(values: DoubleArray): ContiguousVector = ContiguousVector(values)
    }
}
