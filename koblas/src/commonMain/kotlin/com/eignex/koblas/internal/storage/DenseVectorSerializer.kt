package com.eignex.koblas.internal.storage

import com.eignex.koblas.DenseVector
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/** The wire representation contains logical entries in order, without buffer padding or stride. */
@Serializable
@SerialName("DenseVector")
internal class DenseVectorPayload(val values: DoubleArray)

internal object DenseVectorSerializer : KSerializer<DenseVector> {
    private val delegate = DenseVectorPayload.serializer()
    override val descriptor = delegate.descriptor

    override fun serialize(encoder: Encoder, value: DenseVector) {
        val values = if (value.isWholeArray) value.values else value.toDoubleArray()
        delegate.serialize(encoder, DenseVectorPayload(values))
    }

    override fun deserialize(decoder: Decoder): DenseVector = DenseVector.wrap(delegate.deserialize(decoder).values)
}
