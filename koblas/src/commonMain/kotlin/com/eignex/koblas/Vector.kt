package com.eignex.koblas

/** Mutable vector contract with a fixed logical size. */
public interface Vector {
    /** Number of entries, counting the unstored zeros of a sparse vector. */
    public val size: Int

    /** The entry at index (i). Throws `IndexOutOfBoundsException` outside `0 until size`, whatever the storage. */
    public operator fun get(i: Int): Double

    /** Writes [v] at index [i]. Throws [IndexOutOfBoundsException] outside `0 until size`. */
    public operator fun set(i: Int, v: Double)

    /** Materialise into a fresh dense `DoubleArray`, independent of the internal storage. */
    public fun toDoubleArray(): DoubleArray = DoubleArray(size) { get(it) }
}
