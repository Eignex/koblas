package com.eignex.koblas

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SerializationTest {

    private val json = Json

    @Test
    fun `DenseMatrix round-trips through JSON for square rectangular and degenerate shapes`() {
        val cases = listOf(
            DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))),
            DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0, 3.0))), // 1x3
            DenseMatrix(1, 1).also { it[0, 0] = 7.5 },
            DenseMatrix(3, 0), // 3 empty rows survive (cols=0)
            DenseMatrix(0, 0),
        )
        for (m in cases) {
            val back = json.decodeFromString(DenseMatrix.serializer(), json.encodeToString(m))
            assertEquals(m, back, "round-trip $m")
        }
    }

    @Test
    fun `DenseMatrix wire form is a 2D array`() {
        val m = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        assertEquals("[[1.0,2.0],[3.0,4.0]]", json.encodeToString(m))
    }

    @Test
    fun `DenseVector round-trips including empty and singleton`() {
        for (v in listOf(
            DenseVector.of(doubleArrayOf(1.0, -2.0, 3.5)),
            DenseVector(0),
            DenseVector.of(doubleArrayOf(9.0)),
        )) {
            assertEquals(v, json.decodeFromString(DenseVector.serializer(), json.encodeToString(v)))
        }
    }

    @Test
    fun `SparseVector round-trips including empty and unsorted indices`() {
        for (v in listOf(
            SparseVector.of(5, intArrayOf(2, 0), doubleArrayOf(3.0, 1.0)),
            SparseVector.of(4, IntArray(0), DoubleArray(0)),
            SparseVector.of(3, intArrayOf(1), doubleArrayOf(0.0)), // a stored explicit zero
        )) {
            assertEquals(v, json.decodeFromString(SparseVector.serializer(), json.encodeToString(v)))
        }
    }

    // Note: DenseMatrix serializes to a bare 2D JSON array (the readable, stable wire form), which cannot
    // carry a polymorphic type discriminator — so MatrixView is serialized via its concrete DenseMatrix
    // type (above), not polymorphically. VectorView's object subtypes do support polymorphism (below).

    @Test
    fun `VectorView round-trips polymorphically preserving dense and sparse types`() {
        val dense: VectorView = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val sparse: VectorView = SparseVector.of(4, intArrayOf(0, 3), doubleArrayOf(1.0, 2.0))
        val backDense = json.decodeFromString(
            VectorView.serializer(),
            json.encodeToString(VectorView.serializer(), dense),
        )
        val backSparse =
            json.decodeFromString(VectorView.serializer(), json.encodeToString(VectorView.serializer(), sparse))
        assertTrue(backDense is DenseVector)
        assertTrue(backSparse is SparseVector)
        assertEquals(dense, backDense)
        assertEquals(sparse, backSparse)
    }
}
