package com.eignex.koblas.internal.storage

import com.eignex.koblas.DenseMatrix
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/** The wire representation contains logical entries only, never padding or neighboring matrices. */
@Serializable
@SerialName("DenseMatrix")
internal class DenseMatrixPayload(val rows: Int, val cols: Int, val values: DoubleArray)

internal object DenseMatrixSerializer : KSerializer<DenseMatrix> {
    private val delegate = DenseMatrixPayload.serializer()
    override val descriptor = delegate.descriptor

    override fun serialize(encoder: Encoder, value: DenseMatrix) {
        val values = if (value.isWholeBuffer) value.values else (copyMatrixStorage(value) as DenseMatrix).values
        delegate.serialize(encoder, DenseMatrixPayload(value.rows, value.cols, values))
    }

    override fun deserialize(decoder: Decoder): DenseMatrix {
        val payload = delegate.deserialize(decoder)
        requireShape(payload.values.size.toLong() == payload.rows.toLong() * payload.cols) {
            "serialized values must exactly match the matrix shape"
        }
        return DenseMatrix.wrap(payload.rows, payload.cols, payload.values)
    }
}
