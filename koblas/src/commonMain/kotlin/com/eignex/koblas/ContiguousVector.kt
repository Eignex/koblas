package com.eignex.koblas

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * A dense vector that owns its whole buffer, so its entries are exactly [values], in order and complete.
 *
 * @property values the flat backing array. The vector is mutable through it and [set]; do not use the vector as
 *   a hash-map key while mutating it.
 */
@Serializable
@SerialName("ContiguousVector")
public class ContiguousVector internal constructor(public override val values: DoubleArray) :
    VectorStorage,
    DenseVector {
    internal constructor(size: Int) : this(DoubleArray(size))

    override val size: Int get() = values.size
    override val offset: Int get() = 0
    override val stride: Int get() = 1

    override fun get(i: Int): Double {
        requireInBounds(i, size)
        return values[i]
    }

    override fun toDoubleArray(): DoubleArray = values.copyOf()

    override fun set(i: Int, v: Double) {
        requireInBounds(i, size)
        values[i] = v
    }

    override fun equals(other: Any?): Boolean =
        this === other || (other is ContiguousVector && values.contentEquals(other.values))
    override fun hashCode(): Int = values.contentHashCode()
    override fun toString(): String = "ContiguousVector(size=$size)"
}
