package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.*

class VectorOperationsTest {
    @Test
    fun `sparse norms use selected kernels over stored values only`() {
        val source = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, 100), doubleArrayOf(2.0, -3.0))
        val calls = mutableListOf<String>()
        val engine = object : KoblasEngine by BuiltinEngines.scalar {
            override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "asum"
                return BuiltinEngines.scalar.asum(v, vOff, len, vStride)
            }
            override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "nrm2"
                return BuiltinEngines.scalar.nrm2(v, vOff, len, vStride)
            }
            override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "iamax"
                return BuiltinEngines.scalar.iamax(v, vOff, len, vStride)
            }
        }

        for (type in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) engine.normVector(source, type)

        assertEquals(listOf("asum", "nrm2", "iamax", "iamax"), calls)
    }

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
            override fun set(i: Int, v: Double) = error("Unexpected mutation")
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
            override fun set(i: Int, v: Double) = error("Unexpected mutation")
            override fun get(i: Int): Double = error("Read opaque storage")
            override fun toDoubleArray(): DoubleArray = error("Snapshotted opaque storage")
        }
        val destination = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        val before = destination.values.copyOf()

        koblas.axpyVector(0.0, source, destination)

        assertContentEquals(before, destination.values)
        assertFailsWith<IllegalArgumentException> { koblas.axpyVector(0.0, source, DenseVector.zero(3)) }
    }

    @Test
    fun `identical mapping axpy agrees with the scalar snapshot reference`() {
        val inputs = doubleArrayOf(
            1.5e200,
            -1.5e200,
            0.75,
            -0.0,
            0.0,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            Double.NaN,
        )
        for (alpha in doubleArrayOf(0.0, -0.0, 0.75, -1.0)) {
            for ((source, destination) in identicalDenseMappings()) {
                for (i in 0 until source.size) source[i] = inputs[i % inputs.size]
                val expected = source.values.copyOf()
                BuiltinEngines.scalar.axpy(
                    expected,
                    destination.offset,
                    alpha,
                    source.toDoubleArray(),
                    0,
                    source.size,
                    destination.stride,
                )

                koblas.axpyVector(alpha, source, destination)

                assertAxpyAgreesWithReference(expected, source.values)
            }
        }
    }

    @Test
    fun `reversed overlapping axpy agrees with the scalar snapshot reference`() {
        val backing = DoubleArray(12) { it + 0.25 }
        val source = StridedVector(backing, 2, 5)
        val destination = StridedVector(backing, 6, 5, -1)
        val expected = backing.copyOf()
        BuiltinEngines.scalar.axpy(expected, 6, 0.75, source.toDoubleArray(), 0, 5, -1)

        koblas.axpyVector(0.75, source, destination)

        assertAxpyAgreesWithReference(expected, backing)
    }

    @Test
    fun `sparse axpy accepts a prefix view and preserves its padding`() {
        for (size in intArrayOf(0, 3)) {
            val backing = DoubleArray(7) { it + 0.5 }
            val destination = StridedVector(backing, 0, size)
            val source = if (size == 0) sparse(0) else sparse(size, 0 to 2.0, 2 to -1.0)
            val expected = backing.copyOf()
            BuiltinEngines.scalar.axpy(source.indices, 0, source.values, 0, source.indices.size, 2.0, expected)

            koblas.axpyVector(2.0, source, destination)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `axpy adds alpha scaled x to y for any sparsity`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        koblas.axpyVector(2.0, sparse(3, 0 to 1.0, 2 to -1.0), y)
        assertEquals(dense(3.0, 2.0, 1.0), y)
    }

    @Test
    fun `scale mutates in place and respects identity`() {
        val v = DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        koblas.scale(v.values, v.offset, 0.5, v.size, v.stride)
        assertEquals(dense(0.5, -1.0, 1.5), v)
        koblas.scale(v.values, v.offset, 1.0, v.size, v.stride)
        assertEquals(dense(0.5, -1.0, 1.5), v)
    }

    @Test
    fun `vector operations reject mismatched sizes`() {
        assertFailsWith<DimensionMismatch> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<DimensionMismatch> { koblas.axpyVector(1.0, dense(1.0, 2.0), dense(1.0)) }
    }

    @Test
    fun `axpy snapshots a foreign source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = if (borrowed) {
                doubleArrayOf(9.0, 1.0, 2.0, 3.0, 4.0, 9.0)
            } else {
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            }
            val destination = if (borrowed) StridedVector(backing, 1, 4) else DenseVector.wrap(backing)
            val source = ForeignVector(StridedVector(backing, destination.offset + 3, 4, -1))
            val expected = backing.copyOf()
            BuiltinEngines.scalar.axpy(expected, destination.offset, 2.0, source.toDoubleArray(), 0, 4)

            koblas.axpyVector(2.0, source, destination)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `axpy preserves a borrowed source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = StridedVector(backing, 3, 4, -1)
            val expected = DoubleArray(backing.size) { backing[it] + 2.0 * source[it] }

            if (borrowed) {
                koblas.axpyVector(2.0, source, StridedVector(backing, 0, 4))
            } else {
                koblas.axpyVector(2.0, source, DenseVector.wrap(backing))
            }

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `borrowed axpy with sparse input leaves unstored positions untouched`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0)
        val source = sparse(3, 1 to 2.0)
        val destination = StridedVector(backing, 2, 3, -1)
        val expected = DenseVector.of(destination.toDoubleArray())
        koblas.axpyVector(Double.POSITIVE_INFINITY, source, expected)

        koblas.axpyVector(Double.POSITIVE_INFINITY, source, destination)

        assertContentEquals(expected.values.reversedArray(), backing)
    }

    @Test
    fun `axpy accepts a foreign source`() {
        val source = foreignRamp(6)
        val destination = DenseVector.zero(6)
        val expected = DoubleArray(6) { 2.0 * source[it] }

        koblas.axpyVector(2.0, source, destination)

        assertClose(expected, destination.values, "foreign axpy")
    }

    @Test
    fun `scaling a borrowed slice leaves other entries untouched`() {
        val backing = DoubleArray(8) { it + 1.0 }
        val even = StridedVector(backing, 0, 4, 2)
        val expected = backing.copyOf()
        ScalarEngine.scale(expected, 0, 2.0, 4, 2)

        koblas.scale(even.values, even.offset, 2.0, even.size, even.stride)

        assertContentEquals(expected, backing)
    }

    @Test
    fun `alpha zero leaves the destination unchanged`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))

        koblas.axpyVector(0.0, DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0)), y)

        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0), y.values)
    }
    private class ForeignVector(private val backing: DenseVector) : Vector {
        override val size: Int get() = backing.size
        override fun get(i: Int): Double = backing[i]
        override fun set(i: Int, v: Double) {
            backing[i] = v
        }
    }

    private fun foreignRamp(size: Int) = ForeignVector(DenseVector.of(DoubleArray(size) { it * 0.5 - 1.0 }))
    private fun dense(vararg values: Double) = DenseVector.of(values)
    private fun sparse(size: Int, vararg entries: Pair<Int, Double>) = SparseVector.of(
        size,
        entries.map { it.first }.toIntArray(),
        entries.map { it.second }.toDoubleArray(),
    )
    private fun assertAxpyAgreesWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            when {
                expected[i].isNaN() -> assertTrue(actual[i].isNaN())
                expected[i] == 0.0 -> assertEquals(expected[i].toBits(), actual[i].toBits())
                else -> assertClose(expected[i], actual[i], "axpy backing index $i")
            }
        }
    }

    private fun identicalDenseMappings(): List<Pair<DenseVector, DenseVector>> = buildList {
        for (size in intArrayOf(0, 1, 5, 129)) {
            for (sameWrapper in booleanArrayOf(false, true)) {
                val whole = DenseVector.wrap(DoubleArray(size) { it + 0.25 })
                add(whole to if (sameWrapper) whole else whole.view(0, whole.size))
                for (stride in intArrayOf(1, 3, -1, -3)) {
                    val backing = DoubleArray(3 * size + 4) { it + 0.25 }
                    val offset = if (stride > 0 || size == 0) 2 else 2 + (size - 1) * -stride
                    val source = StridedVector(backing, offset, size, stride)
                    add(source to if (sameWrapper) source else source.view(0, source.size))
                }
            }
        }
    }
}
