package com.eignex.koblas

/**
 * A dense vector backed by a contiguous [data] array. A thin value wrapper: linear-algebra operations
 * live on [LinearAlgebra] (and take raw [DoubleArray]s) so a real BLAS backend sees plain buffers.
 *
 * @property data the vector elements.
 */
class Vector(val data: DoubleArray) {
    /** Number of elements. */
    val size: Int get() = data.size

    /** Element [i]. */
    operator fun get(i: Int): Double = data[i]

    /** Set element [i]. */
    operator fun set(i: Int, value: Double) {
        data[i] = value
    }

    override fun equals(other: Any?): Boolean = this === other || (other is Vector && data.contentEquals(other.data))

    override fun hashCode(): Int = data.contentHashCode()

    override fun toString(): String = "Vector(size=$size)"

    /** Factories for common vectors. */
    companion object {
        /** A zero vector of length [size]. */
        fun zeros(size: Int): Vector = Vector(DoubleArray(size))
    }
}
