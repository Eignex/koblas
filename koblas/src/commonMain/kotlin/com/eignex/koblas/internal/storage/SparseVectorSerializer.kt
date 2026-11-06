package com.eignex.koblas.internal.storage

import com.eignex.koblas.SparseVector
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/** Sparse views encode only their own support, with the same field names as owning vectors. */
@Serializable
@SerialName("SparseVector")
internal class SparseVectorPayload(
    val size: Int,
    @SerialName("indices") val indices: IntArray,
    @SerialName("values") val values: DoubleArray,
)

internal object SparseVectorSerializer : KSerializer<SparseVector> {
    private val delegate = SparseVectorPayload.serializer()
    override val descriptor = delegate.descriptor

    override fun serialize(encoder: Encoder, value: SparseVector) {
        val start = value.offset
        val whole = start == 0 && value.nnz == value.values.size
        val indices = if (whole) value.indices else value.copyIndices()
        val values = if (whole) value.values else value.values.copyOfRange(start, start + value.nnz)
        delegate.serialize(encoder, SparseVectorPayload(value.size, indices, values))
    }

    override fun deserialize(decoder: Decoder): SparseVector {
        val payload = delegate.deserialize(decoder)
        return SparseVector.wrap(payload.size, payload.indices, payload.values)
    }
}
