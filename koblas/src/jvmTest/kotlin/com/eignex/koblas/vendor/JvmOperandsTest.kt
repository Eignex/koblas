package com.eignex.koblas.vendor

import com.eignex.koblas.StridedVector
import com.eignex.koblas.internal.jvm25.stage
import java.lang.foreign.Arena
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class JvmOperandsTest {
    @Test
    fun `native allocation counts entries instead of the physical span`() {
        val storage = DoubleArray(2_000_005)
        for (stride in listOf(1_000_000, -1_000_000)) {
            val vector = StridedVector(storage, if (stride > 0) 2 else 2_000_002, 3, stride)

            Arena.ofConfined().use { arena ->
                val native = arena.stage(vector)

                assertEquals(3L * JAVA_DOUBLE.byteSize(), native.segment.byteSize(), "stride $stride")
            }
        }
    }

    @Test
    fun `staging gathers entries in physical order and preserves traversal direction`() {
        for (stride in listOf(1, -1, 3, -3)) {
            val step = abs(stride)
            val storage = DoubleArray(2 * step + 5) { Double.NaN }
            val entries = doubleArrayOf(2.0, -5.0, 7.0)
            entries.forEachIndexed { index, value -> storage[2 + index * step] = value }
            val vector = StridedVector(storage, if (stride > 0) 2 else 2 + 2 * step, entries.size, stride)

            Arena.ofConfined().use { arena ->
                val native = arena.stage(vector)

                assertContentEquals(entries, native.segment.toArray(JAVA_DOUBLE), "stride $stride")
                assertEquals(if (stride > 0) 1 else -1, native.increment, "stride $stride")
            }
        }
    }

    @Test
    fun `writeback scatters packed entries without changing gaps or padding`() {
        for (stride in listOf(1, -1, 3, -3)) {
            val step = abs(stride)
            val storage = DoubleArray(2 * step + 5) { it.toDouble() }
            val expected = storage.copyOf()
            val entries = doubleArrayOf(20.0, -50.0, 70.0)
            entries.forEachIndexed { index, value -> expected[2 + index * step] = value }
            val vector = StridedVector(storage, if (stride > 0) 2 else 2 + 2 * step, entries.size, stride)

            Arena.ofConfined().use { arena ->
                val native = arena.stage(vector)
                entries.forEachIndexed { index, value -> native.segment.setAtIndex(JAVA_DOUBLE, index.toLong(), value) }
                native.writeBack()
            }

            assertContentEquals(expected, storage, "stride $stride")
        }
    }

    @Test
    fun `interleaved writebacks preserve both staged results`() {
        val storage = DoubleArray(12) { it.toDouble() }
        val expected = storage.copyOf()
        val first = StridedVector(storage, 1, 3, 3)
        val second = StridedVector(storage, 8, 3, -3)
        val firstEntries = doubleArrayOf(20.0, 30.0, 40.0)
        val secondEntries = doubleArrayOf(50.0, 60.0, 70.0)
        for (index in firstEntries.indices) {
            expected[1 + index * 3] = firstEntries[index]
            expected[2 + index * 3] = secondEntries[index]
        }

        Arena.ofConfined().use { arena ->
            val nativeFirst = arena.stage(first)
            val nativeSecond = arena.stage(second)
            for (index in firstEntries.indices) {
                nativeFirst.segment.setAtIndex(JAVA_DOUBLE, index.toLong(), firstEntries[index])
                nativeSecond.segment.setAtIndex(JAVA_DOUBLE, index.toLong(), secondEntries[index])
            }
            nativeFirst.writeBack()
            nativeSecond.writeBack()
        }

        assertContentEquals(expected, storage)
    }

    @Test
    fun `singletons normalize untraversed strides and write only their entry`() {
        for (stride in listOf(1, -1, Int.MAX_VALUE, Int.MIN_VALUE)) {
            val storage = doubleArrayOf(2.0, 3.0, 4.0)
            val vector = StridedVector(storage, 1, 1, stride)

            Arena.ofConfined().use { arena ->
                val native = arena.stage(vector)
                assertEquals(JAVA_DOUBLE.byteSize(), native.segment.byteSize(), "stride $stride")
                assertEquals(1, native.increment, "stride $stride")
                assertEquals(3.0, native.segment.getAtIndex(JAVA_DOUBLE, 0L), "stride $stride")
                native.segment.setAtIndex(JAVA_DOUBLE, 0L, 5.0)
                native.writeBack()
            }

            assertContentEquals(doubleArrayOf(2.0, 5.0, 4.0), storage, "stride $stride")
        }
    }

    @Test
    fun `empty staging keeps a dummy allocation without touching storage`() {
        for (stride in listOf(1, -1, Int.MAX_VALUE, Int.MIN_VALUE)) {
            val storage = doubleArrayOf(2.0, 3.0, 4.0)
            val expected = storage.copyOf()
            val vector = StridedVector(storage, storage.size, 0, stride)

            Arena.ofConfined().use { arena ->
                val native = arena.stage(vector)
                assertEquals(JAVA_DOUBLE.byteSize(), native.segment.byteSize(), "stride $stride")
                native.segment.setAtIndex(JAVA_DOUBLE, 0L, 5.0)
                native.writeBack()
            }

            assertContentEquals(expected, storage, "stride $stride")
        }
    }
}
