package com.eignex.koblas

import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonException
import kotlin.test.*

class SparseVectorTest {
    @Test
    fun `writes through the vector contract insert sorted entries and keep zeros`() {
        val vector = SparseVector.wrap(6, intArrayOf(1, 4), doubleArrayOf(2.0, -1.0))
        val mutable: Vector = vector

        mutable[5] = 0.0
        mutable[0] = -3.0
        mutable[3] = 8.0
        val indices = vector.indices
        val values = vector.values
        mutable[4] = 0.0

        assertContentEquals(intArrayOf(0, 1, 3, 4, 5), vector.indices)
        assertContentEquals(doubleArrayOf(-3.0, 2.0, 8.0, 0.0, 0.0), vector.values)
        assertSame(indices, vector.indices)
        assertSame(values, vector.values)
        assertEquals(vector, Json.decodeFromString(SparseVector.serializer(), Json.encodeToString(vector)))
        assertFailsWith<IndexOutOfBoundsException> { mutable[-1] = 1.0 }
        assertFailsWith<IndexOutOfBoundsException> { mutable[6] = 1.0 }
    }

    @Test
    fun `factories reject negative sizes`() {
        assertFailsWith<DimensionMismatch> { SparseVector.wrap(-5, IntArray(0), DoubleArray(0)) }
        assertFailsWith<DimensionMismatch> { SparseVector.of(-3, IntArray(0), DoubleArray(0)) }
    }

    @Test
    fun `of rejects indices and values of different lengths`() {
        assertFailsWith<DimensionMismatch> {
            SparseVector.of(3, intArrayOf(0), doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun `of copies its inputs`() {
        val idx = intArrayOf(0, 2)
        val vals = doubleArrayOf(1.0, 3.0)
        val v = SparseVector.of(5, idx, vals)
        idx[0] = 4
        vals[1] = 99.0
        assertEquals(1.0, v[0])
        assertEquals(3.0, v[2])
    }

    @Test
    fun `toDoubleArray materialises stored entries`() {
        val v = SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        assertTrue(v.toDoubleArray().contentEquals(doubleArrayOf(0.0, 2.0, 0.0, 4.0)))
    }

    @Test
    fun `equals and hashCode respect content`() {
        val a = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val b = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        val different = SparseVector.of(4, intArrayOf(0, 2), doubleArrayOf(1.0, 4.0))
        val sizeDiff = SparseVector.of(5, intArrayOf(0, 2), doubleArrayOf(1.0, 3.0))
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, different)
        assertNotEquals(a, sizeDiff)
        assertNotEquals<Any?>(a, "x")
    }

    @Test
    fun `of sorts by index and sums duplicates`() {
        val v = SparseVector.of(5, intArrayOf(3, 0, 3), doubleArrayOf(1.0, 2.0, 0.5))
        assertTrue(intArrayOf(0, 3).contentEquals(v.indices), "indices should be ascending: ${v.indices.toList()}")
        assertTrue(doubleArrayOf(2.0, 1.5).contentEquals(v.values), "duplicate at 3 should sum to 1.5")
        assertEquals(2.0, v[0])
        assertEquals(1.5, v[3])
        assertEquals(0.0, v[1])
        val seen = ArrayList<Int>()
        v.forEachStored { i, _ -> seen.add(i) }
        assertEquals(listOf(0, 3), seen)
    }

    @Test
    fun `get finds every stored entry by binary search`() {
        val idx = intArrayOf(0, 1, 4, 7, 8, 15)
        val v = SparseVector.of(16, idx, DoubleArray(idx.size) { it + 1.0 })
        for ((k, i) in idx.withIndex()) assertEquals(k + 1.0, v[i], "missed stored index $i")
        for (i in 0 until 16) if (i !in idx) assertEquals(0.0, v[i], "index $i should be absent")
    }

    @Test
    fun `copyIndices cannot mutate sparse structure`() {
        val v = SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        val indices = v.copyIndices()

        indices[0] = 0

        assertEquals(2.0, v[1])
        assertEquals(0.0, v[0])
    }

    @Test
    fun `wrap adopts ascending arrays and validates them`() {
        val indices = intArrayOf(0, 3)
        val values = doubleArrayOf(2.0, 1.5)
        val v = SparseVector.wrap(5, indices, values)
        assertEquals(2.0, v[0])
        assertEquals(1.5, v[3])
        assertEquals(0.0, v[1])
        assertSame(indices, v.indices)
        assertSame(values, v.values)
        assertFailsWith<IllegalArgumentException> {
            SparseVector.wrap(5, intArrayOf(3, 0), doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<DimensionMismatch> {
            SparseVector.wrap(5, intArrayOf(0, 1), doubleArrayOf(1.0))
        }
        assertFailsWith<IndexOutOfBoundsException> {
            SparseVector.wrap(5, intArrayOf(-1), doubleArrayOf(1.0))
        }
        assertFailsWith<IndexOutOfBoundsException> {
            SparseVector.wrap(5, intArrayOf(5), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException> {
            SparseVector.wrap(5, intArrayOf(1, 1), doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun `round trips through JSON including empty and stored zero cases`() {
        for (v in listOf(
            SparseVector.of(5, intArrayOf(2, 0), doubleArrayOf(3.0, 1.0)), // of sorts these on the way in
            SparseVector.of(4, IntArray(0), DoubleArray(0)),
            SparseVector.of(3, intArrayOf(1), doubleArrayOf(0.0)), // a stored explicit zero
        )) {
            assertEquals(v, Json.decodeFromString(SparseVector.serializer(), Json.encodeToString(v)))
        }
    }

    @OptIn(ExperimentalSerializationApi::class)
    @Test
    fun `deserialization rejects a negative size`() {
        val failure = assertFailsWith<JsonException> {
            Json.decodeFromString<VectorStorage>(
                """{"type":"com.eignex.koblas.SparseVector","size":-5,"storedIndices":[],"storedValues":[]}""",
            )
        }
        assertIs<DimensionMismatch>(failure.cause)
    }
}
