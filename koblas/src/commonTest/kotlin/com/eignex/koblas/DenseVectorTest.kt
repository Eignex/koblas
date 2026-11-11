package com.eignex.koblas

import com.eignex.koblas.internal.storage.dense.*
import kotlinx.serialization.json.*
import kotlin.test.*

class DenseVectorTest {
    @Test
    fun `zero builds a zero filled vector`() {
        val z = DenseVector.zero(3)
        assertEquals(3, z.size)
        for (i in 0 until 3) assertEquals(0.0, z[i])
    }

    @Test
    fun `zero rejects a negative size`() {
        assertFailsWith<DimensionMismatch> { DenseVector.zero(-3) }
    }

    @Test
    fun `of copies its input`() {
        val src = doubleArrayOf(1.0, 2.0, 3.0)
        val v = DenseVector.of(src)
        src[0] = 99.0
        assertEquals(1.0, v[0])
    }

    @Test
    fun `wrap defaults to the remaining unit stride buffer`() {
        val values = doubleArrayOf(1.0, 2.0, 3.0, 4.0)

        val vector = DenseVector.wrap(values, offset = 2)

        assertSame(values, vector.values)
        assertEquals(2, vector.size)
        assertEquals(1, vector.stride)
        assertContentEquals(doubleArrayOf(3.0, 4.0), vector.toDoubleArray())
    }

    @Test
    fun `equality and hashing compare logical entries across strides`() {
        val values = doubleArrayOf(1.5, Double.NaN, -0.0, 2.0)
        val compact = DenseVector.of(values)
        for (stride in intArrayOf(-3, -1, 1, 3)) {
            val backing = DoubleArray(14) { Double.NaN }
            val offset = if (stride > 0) 2 else 2 - (values.size - 1) * stride
            val view = DenseVector.wrap(backing, offset, values.size, stride)
            for (i in values.indices) view[i] = values[i]

            assertEquals(compact, view)
            assertEquals(view, compact)
            assertEquals(compact.hashCode(), view.hashCode())
            backing[0] = 42.0
            assertEquals(compact, view)
            view[2] = 0.0
            assertNotEquals(compact, view)
        }
    }

    @Test
    fun `serialization writes logical entries and decodes independent contiguous storage`() {
        val values = doubleArrayOf(1.0, -2.0, 3.0)
        for (stride in intArrayOf(-2, 1, 2)) {
            val backing = DoubleArray(9) { Double.NaN }
            val offset = if (stride > 0) 2 else 2 - (values.size - 1) * stride
            val view = DenseVector.wrap(backing, offset, values.size, stride)
            for (i in values.indices) view[i] = values[i]

            val encoded = Json.encodeToString(DenseVector.serializer(), view)
            val decoded = Json.decodeFromString(DenseVector.serializer(), encoded)
            val payload = Json.parseToJsonElement(encoded).jsonObject

            assertEquals(setOf("values"), payload.keys)
            assertEquals(values.toList(), payload.getValue("values").jsonArray.map { it.jsonPrimitive.double })
            assertEquals(view, decoded)
            assertEquals(0, decoded.offset)
            assertEquals(1, decoded.stride)
            assertEquals(values.size, decoded.values.size)
            assertNotSame(backing, decoded.values)
            decoded[0] = 42.0
            assertEquals(1.0, view[0])
        }
    }

    @Test
    fun `reversed views round trip through polymorphic vector storage`() {
        val source: VectorStorage = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0)).view(3, 2, -2)

        val encoded = Json.encodeToString(VectorStorage.serializer(), source)
        val decoded = Json.decodeFromString(VectorStorage.serializer(), encoded)

        assertEquals("DenseVector", Json.parseToJsonElement(encoded).jsonObject.getValue("type").jsonPrimitive.content)
        assertEquals(source, decoded)
        assertContentEquals(doubleArrayOf(4.0, 2.0), assertIs<DenseVector>(decoded).values)
    }

    @Test
    fun `empty views compare and serialize without reading the buffer`() {
        for (stride in intArrayOf(Int.MIN_VALUE, -1, 1, Int.MAX_VALUE)) {
            val source = DenseVector.wrap(doubleArrayOf(Double.NaN), 1, 0, stride)

            assertEquals(DenseVector.zero(0), source)
            assertEquals(DenseVector.zero(0).hashCode(), source.hashCode())
            val encoded = Json.encodeToString(DenseVector.serializer(), source)
            val decoded = Json.decodeFromString(DenseVector.serializer(), encoded)

            assertEquals(source, decoded)
            assertTrue(decoded.values.isEmpty())
        }
    }

    @Test
    fun `empty and extreme raw layouts are validated before use`() {
        val values = DoubleArray(3)
        for (create in listOf<() -> DenseVector>(
            { DenseVector.wrap(values, -1, 0) },
            { DenseVector.wrap(values, 4, 0) },
            { DenseVector.wrap(values, 0, -1) },
            { DenseVector.wrap(values, Int.MIN_VALUE, Int.MAX_VALUE, Int.MAX_VALUE) },
            { DenseVector.wrap(values, Int.MAX_VALUE, Int.MAX_VALUE, Int.MIN_VALUE) },
        )) {
            assertFailsWith<DimensionMismatch> { create() }
        }
    }

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
            assertEquals(v, Json.decodeFromString(DenseVector.serializer(), Json.encodeToString(v)))
        }
    }

    // Strides of equal and unequal magnitude in both directions, origins inside and at the end of the buffer,
    // and empty, singleton and longer views reach every branch of the overlap test.
    @Test
    fun `overlap agrees with physical index intersection`() {
        val backing = DoubleArray(8)
        val views = buildList {
            for (stride in intArrayOf(-2, -1, 1, 2, 3)) {
                for (offset in intArrayOf(0, 1, 4, backing.size)) {
                    for (size in intArrayOf(0, 1, 2, 4)) {
                        val last = offset + (size - 1) * stride
                        if (size == 0 || (offset in backing.indices && last in backing.indices)) {
                            add(DenseVector.wrap(backing, offset, size, stride))
                        }
                    }
                }
            }
        }
        val addressed = views.map { view -> (0 until view.size).map { view.offset + it * view.stride }.toSet() }

        for (i in views.indices) {
            for (j in views.indices) {
                val expected = addressed[i].any { it in addressed[j] }

                if (views[i].overlaps(views[j]) != expected) fail("${views[i]} against ${views[j]}: expected $expected")
            }
        }
    }

    @Test
    fun `overlap handles extreme singleton strides and independent buffers`() {
        val backing = DoubleArray(2)
        for (stride in intArrayOf(Int.MIN_VALUE, Int.MAX_VALUE)) {
            val singleton = DenseVector.wrap(backing, 0, 1, stride)

            assertEquals(true, singleton.overlaps(DenseVector.wrap(backing, 0, 2)))
            assertEquals(false, singleton.overlaps(DenseVector.wrap(backing, 1, 1, stride)))
            assertEquals(false, singleton.overlaps(DenseVector.wrap(backing, 0, 0, stride)))
            assertEquals(false, singleton.overlaps(DenseVector.wrap(backing.copyOf(), 0, 1, stride)))
        }
    }

    @Test
    fun `nested views cannot escape the logical parent`() {
        val parent = DenseVector.wrap(DoubleArray(12), 4, 3)

        for ((offset, size, stride) in listOf(Triple(-1, 1, 1), Triple(3, 1, 1), Triple(1, 3, 1), Triple(1, 3, -1))) {
            assertFailsWith<DimensionMismatch> { parent.view(offset, size, stride) }
        }
    }

    @Test
    fun `nested view offset arithmetic cannot wrap into the buffer`() {
        val parent = DenseVector.wrap(DoubleArray(3), 0, 2, 2)

        assertFailsWith<DimensionMismatch> { parent.view(Int.MIN_VALUE, 1) }
    }

    @Test
    fun `nested view stride arithmetic cannot wrap into another stride`() {
        val parent = DenseVector.wrap(DoubleArray(1), 0, 1, Int.MAX_VALUE)

        assertFailsWith<DimensionMismatch> { parent.view(0, 1, 3) }
    }

    @Test
    fun `empty nested views accept both logical endpoints`() {
        for (stride in intArrayOf(2, -2)) {
            val parent = DenseVector.wrap(DoubleArray(5), if (stride < 0) 4 else 0, 3, stride)
            for (offset in intArrayOf(0, parent.size)) {
                val empty = parent.view(offset, 0)

                assertEquals(0, empty.size)
                assertContentEquals(doubleArrayOf(), empty.toDoubleArray())
            }
            assertFailsWith<DimensionMismatch> { parent.view(-1, 0) }
            assertFailsWith<DimensionMismatch> { parent.view(parent.size + 1, 0) }
        }
    }

    @Test
    fun `nested views reject negative sizes and zero strides`() {
        val parent = DenseVector.zero(3)

        assertFailsWith<DimensionMismatch> { parent.view(0, -1) }
        assertFailsWith<IllegalArgumentException> { parent.view(0, 1, 0) }
    }

    @Test
    fun `a negative stride presents a reversed borrowed slice`() {
        val vector = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        val reversed = vector.view(offset = 4, size = 3, stride = -2)
        reversed[1] = -3.0

        assertContentEquals(doubleArrayOf(5.0, -3.0, 1.0), reversed.toDoubleArray())
        assertContentEquals(doubleArrayOf(1.0, 2.0, -3.0, 4.0, 5.0), vector.values)
    }

    @Test
    fun `view construction rejects addresses outside the buffer`() {
        assertFailsWith<DimensionMismatch> { DenseVector.wrap(DoubleArray(3), 2, 2) }
        assertFailsWith<IllegalArgumentException> { DenseVector.wrap(DoubleArray(3), 0, 2, 0) }
    }

    // Building the result from the buffer alone would move the window silently: every entry read would still
    // be in range and no call would fail, the answer would just be about different numbers.
    @Test
    fun `a view of a strided vector composes with the window it already had`() {
        val backing = DoubleArray(12) { it.toDouble() }
        val outer: DenseVector = DenseVector.wrap(backing, offset = 2, size = 4, stride = 2)

        assertContentEquals(doubleArrayOf(2.0, 4.0, 6.0, 8.0), outer.view(0, outer.size).toDoubleArray())
        assertContentEquals(doubleArrayOf(4.0, 8.0), outer.view(offset = 1, size = 2, stride = 2).toDoubleArray())
    }
}
