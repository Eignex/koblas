package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.engine.BuiltinEngines
import kotlin.test.*

class VectorOperationsTest {
    @Test
    fun `sparse sum uses the selected kernel over stored values only`() {
        val source = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, 100), doubleArrayOf(2.0, -3.0))
        var length = -1
        val engine = object : KoblasEngine by BuiltinEngines.scalar {
            override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                length = len
                return BuiltinEngines.scalar.sum(v, vOff, len, vStride)
            }
        }
        val expected = BuiltinEngines.scalar.sum(source.values, 0, source.values.size)

        val actual = engine.sumVector(source)

        assertEquals(source.values.size, length)
        assertClose(expected, actual, "sparse sum")
    }

    @Test
    fun `sparse dot reads only its support from opaque storage`() {
        val source = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, 100), doubleArrayOf(2.0, -3.0))
        val reads = mutableListOf<Int>()
        val other = object : Vector {
            override val size = source.size
            override fun get(i: Int): Double {
                reads += i
                return when (i) {
                    1 -> 4.0
                    100 -> -2.0
                    else -> error("Read outside sparse support")
                }
            }
            override fun toDoubleArray(): DoubleArray = error("Materialized opaque storage")
        }
        val expected = BuiltinEngines.scalar.dot(source.values, 0, doubleArrayOf(4.0, -2.0), 0, 2)

        for ((x, y) in listOf(source to other, other to source)) {
            reads.clear()

            val actual = BuiltinEngines.scalar.dotVector(x, y)

            assertClose(expected, actual, "sparse dot")
            assertEquals(source.indices.toList(), reads)
        }
    }

    @Test
    fun `sparse writes into strided views agree with the scalar reference over the whole buffer`() {
        val source = SparseVector.wrap(4, intArrayOf(0, 2), doubleArrayOf(2.0, -3.0))
        for (stride in intArrayOf(-3, -1, 1, 3)) {
            for (copy in booleanArrayOf(false, true)) {
                val backing = DoubleArray(16) { it * 0.25 - 2.0 }
                val destination = StridedVector(backing, if (stride < 0) 11 else 2, 4, stride)
                val expected = backing.copyOf()
                val packed = destination.toDoubleArray()
                if (copy) {
                    packed.fill(0.0)
                    BuiltinEngines.scalar.scatter(source.indices, 0, source.values, 0, 2, packed)
                } else {
                    BuiltinEngines.scalar.axpy(source.indices, 0, source.values, 0, 2, -1.5, packed)
                }
                for (i in packed.indices) expected[destination.offset + i * stride] = packed[i]

                if (copy) {
                    koblas.copyVector(source, destination)
                } else {
                    koblas.axpyVector(-1.5, source, destination)
                }

                assertWritesAgreeWithReference(expected, backing)
            }
        }
    }

    private fun assertWritesAgreeWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertContentEquals(expected, actual)
    }

    @Test
    fun `zero axpy validates size without reading opaque storage`() {
        val source = object : Vector {
            override val size = 4
            override fun get(i: Int): Double = error("Read opaque storage")
            override fun toDoubleArray(): DoubleArray = error("Snapshotted opaque storage")
        }
        val destination = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val before = destination.values.copyOf()

        koblas.axpyVector(0.0, source, destination)

        assertContentEquals(before, destination.values)
        assertFailsWith<IllegalArgumentException> { koblas.axpyVector(0.0, source, DenseVector.zero(3)) }
    }
}
