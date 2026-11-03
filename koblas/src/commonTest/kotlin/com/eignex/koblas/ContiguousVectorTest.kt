package com.eignex.koblas

import kotlinx.serialization.json.Json
import kotlin.test.*

class ContiguousVectorTest {
    @Test
    fun `toDoubleArray returns a copy`() {
        val v = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val copy = v.toDoubleArray()
        copy[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `equals respects content`() {
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val c = DenseVector.of(doubleArrayOf(1.0, 3.0))
        assertEquals(a, b)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, c)
        assertNotEquals<Any?>(a, "not-a-vector")
    }

    @Test
    fun `round trips through JSON including empty and singleton`() {
        for (v in listOf(
            DenseVector.of(doubleArrayOf(1.0, -2.0, 3.5)),
            DenseVector.zero(0),
            DenseVector.of(doubleArrayOf(9.0)),
        )) {
            assertEquals(v, Json.decodeFromString(ContiguousVector.serializer(), Json.encodeToString(v)))
        }
    }
}
