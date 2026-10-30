package com.eignex.koblas

import kotlinx.serialization.Serializable

/** Read-only vector contract. Anything that only reads a vector should take this. */
public interface Vector {
    /** Number of entries, counting the unstored zeros of a sparse vector. */
    public val size: Int

    /** The entry at index (i). Throws `IndexOutOfBoundsException` outside `0 until size`, whatever the storage. */
    public operator fun get(i: Int): Double

    /** Materialise into a fresh dense `DoubleArray`, independent of the internal storage. */
    public fun toDoubleArray(): DoubleArray
}

/**
 * The vector storages koblas defines and serializes, [ContiguousVector] and [SparseVector].
 *
 * This is about which types koblas owns the definition of, not about which own their buffers: either may be
 * wrapped around an array the caller keeps a reference to.
 *
 * Encoded under the storage's own class name, with the numbers under `values` and any index array named for
 * the axis it indexes, the same as [MatrixStorage]. A reader of one payload can then predict the others, and
 * a name in a payload always resolves to the type that wrote it.
 */
@Serializable
public sealed interface VectorStorage : Vector
