package com.eignex.koblas.internal.storage

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseVector
import com.eignex.koblas.internal.storage.dense.*
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
        val values = if (value.isWholeBuffer) value.values else copyDenseMatrix(value).values
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

/** The wire representation contains logical entries in order, without buffer padding or stride. */
@Serializable
@SerialName("DenseVector")
internal class DenseVectorPayload(val values: DoubleArray)

internal object DenseVectorSerializer : KSerializer<DenseVector> {
    private val delegate = DenseVectorPayload.serializer()
    override val descriptor = delegate.descriptor

    override fun serialize(encoder: Encoder, value: DenseVector) {
        val values = if (value.isWholeBuffer) value.values else value.toDoubleArray()
        delegate.serialize(encoder, DenseVectorPayload(values))
    }

    override fun deserialize(decoder: Decoder): DenseVector = DenseVector.wrap(delegate.deserialize(decoder).values)
}

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
