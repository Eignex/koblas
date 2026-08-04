package com.eignex.koblas

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
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
    fun `DenseMatrix wire form is a nested 2D array`() {
        // Assert the structure (rows of numbers), not the literal string — double formatting is
        // platform-specific (JS vs JVM), but the 2D-array shape is the stable contract.
        val m = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0)))
        val rows = json.parseToJsonElement(json.encodeToString(m)).jsonArray
        assertEquals(2, rows.size)
        assertEquals(listOf(1.0, 2.0), rows[0].jsonArray.map { it.jsonPrimitive.double })
        assertEquals(listOf(3.0, 4.0), rows[1].jsonArray.map { it.jsonPrimitive.double })
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
    fun `SparseVector round-trips including empty and stored-zero cases`() {
        for (v in listOf(
            SparseVector.of(5, intArrayOf(2, 0), doubleArrayOf(3.0, 1.0)), // of sorts these on the way in
            SparseVector.of(4, IntArray(0), DoubleArray(0)),
            SparseVector.of(3, intArrayOf(1), doubleArrayOf(0.0)), // a stored explicit zero
        )) {
            assertEquals(v, json.decodeFromString(SparseVector.serializer(), json.encodeToString(v)))
        }
    }

    // Note: DenseMatrix serializes to a bare 2D JSON array (the readable, stable wire form), which cannot
    // carry a polymorphic type discriminator — so a DenseMatrix must be serialized via its concrete type
    // (above), not through MatrixView. SparseMatrix encodes as an object and does not share the
    // limitation, which makes MatrixView polymorphism work for one subtype and not the other; VectorView
    // has no such split, since both its subtypes encode as objects.

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

    @Test
    fun `SparseMatrix round-trips through its CSC arrays`() {
        for (a in listOf(
            SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 3.0), listOf(1 to 2.0))),
            SparseMatrix(2, 1, intArrayOf(0, 1), intArrayOf(1), doubleArrayOf(0.0)), // a stored zero
            SparseMatrix(0, 0, intArrayOf(0), IntArray(0), DoubleArray(0)),
            SparseMatrix(4, 3, intArrayOf(0, 0, 0, 0), IntArray(0), DoubleArray(0)), // all-empty columns
        )) {
            val back = json.decodeFromString(
                SparseMatrix.serializer(),
                json.encodeToString(SparseMatrix.serializer(), a),
            )
            assertEquals(a, back, "round-trip changed $a")
        }
    }

    /**
     * The shape a sparse matrix takes on the wire is its CSC arrays, not a densified grid.
     *
     * Worth pinning: encoding it like DenseMatrix does — as rows — would cost `rows × cols` for a matrix
     * whose whole purpose is to avoid that, and would silently lose which zeros were stored.
     */
    @Test
    fun `SparseMatrix encodes its structure rather than a dense grid`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 5.0), emptyList()))
        val encoded = json.encodeToString(SparseMatrix.serializer(), a)
        assertTrue(encoded.contains("colPtr"), "expected the CSC arrays in $encoded")
        assertTrue(encoded.contains("rowIdx"), "expected the CSC arrays in $encoded")
        assertTrue("0.0" !in encoded, "a structural zero leaked into the payload: $encoded")
    }

    /**
     * A [SparseMatrix] round-trips through the sealed [MatrixView] root, discriminator and all.
     *
     * A [DenseMatrix] does not, and that asymmetry is a known limitation rather than an oversight here:
     * its wire form is a bare 2D array chosen for readability, and a bare array has nowhere to put the
     * type tag a polymorphic decode needs. So [MatrixView] polymorphism works for the storage whose wire
     * form is an object and fails for the one whose form is an array. Fixing it means changing
     * DenseMatrix's payload shape, which is a wire-format decision, not a serialization bug — see the note
     * above. [VectorView] has no such split, since both its subtypes encode as objects.
     */
    @Test
    fun `SparseMatrix round-trips polymorphically through MatrixView`() {
        val sparse: MatrixView = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 2.0)))
        val encoded = json.encodeToString(MatrixView.serializer(), sparse)
        assertTrue(encoded.contains("SparseMatrix"), "expected a type discriminator in $encoded")
        val back = json.decodeFromString(MatrixView.serializer(), encoded)
        assertTrue(back is SparseMatrix, "sparse storage was not preserved")
        assertEquals(sparse, back)
    }
}
