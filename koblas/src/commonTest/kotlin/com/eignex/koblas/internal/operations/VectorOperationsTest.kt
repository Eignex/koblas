package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.*

class VectorOperationsTest {
    @Test
    fun `public engine wrappers support vector convenience adapters`() {
        val engine = object : KoblasEngine by KoblasEngine.scalar {}
        val operands = listOf<Vector>(dense(1.0, -2.0, 3.0), sparse(3, 0 to 1.0, 2 to 3.0))

        for (source in operands) {
            for (other in operands) {
                assertWrapperAgreesWithReference(
                    KoblasEngine.scalar.dotVector(source, other),
                    engine.dotVector(source, other),
                )
            }
            for (norm in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) {
                assertWrapperAgreesWithReference(
                    KoblasEngine.scalar.normVector(source, norm),
                    engine.normVector(source, norm),
                )
            }
            assertEquals(KoblasEngine.scalar.sumVector(source), engine.sumVector(source))
            assertEquals(KoblasEngine.scalar.absoluteSumVector(source), engine.absoluteSumVector(source))
        }
    }

    @Test
    fun `public engine wrappers copy opaque aliases through staged typed operations`() {
        val engine = object : KoblasEngine by KoblasEngine.scalar {}
        val backing = doubleArrayOf(1.0, -2.0, 3.0, 4.0, 5.0)
        val source = ForeignVector(DenseVector.wrap(backing, 0, 3))
        val destination = DenseVector.wrap(backing, 2, 3)
        val expected = backing.copyOf()
        KoblasEngine.scalar.copy(DenseVector.wrap(expected, 0, 3), DenseVector.wrap(expected, 2, 3))

        engine.copyVector(source, destination)

        assertContentEquals(expected, backing)
    }

    @Test
    fun `scale agrees with the scalar reference across storage`() {
        val values = doubleArrayOf(1.5, 0.0, -2.0, 3.25)
        for (alpha in doubleArrayOf(-2.0, 0.0, 1.0)) {
            for (destination in mutationOperands(values)) {
                val expected = values.copyOf()
                KoblasEngine.scalar.kernels.scale(expected, 0, alpha, expected.size)

                koblas.scaleVector(alpha, destination)

                assertClose(expected, destination.toDoubleArray(), "vector scale")
            }
        }
    }

    @Test
    fun `axpy agrees with the scalar reference across source and destination storage`() {
        val x = doubleArrayOf(1.5, 0.0, -2.0, 3.25)
        val y = doubleArrayOf(0.0, 2.5, 1.0, -1.5)
        for (source in mutationOperands(x)) {
            for (destination in mutationOperands(y)) {
                val expected = y.copyOf()
                KoblasEngine.scalar.kernels.axpy(expected, 0, -1.5, x, 0, x.size)

                koblas.axpyVector(-1.5, source, destination)

                assertAxpyAgreesWithReference(expected, destination.toDoubleArray())
            }
        }
    }

    @Test
    fun `axpy snapshots sources sharing opaque destination storage`() {
        for (sparseSource in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val destination: Vector = ForeignVector(DenseVector.wrap(backing, 3, 4, -1))
            val source: Vector = if (sparseSource) {
                SparseVector.wrap(4, intArrayOf(0, 1, 2, 3), backing)
            } else {
                DenseVector.wrap(backing)
            }
            val expected = backing.copyOf()
            KoblasEngine.scalar.kernels.axpy(expected, 3, 2.0, source.toDoubleArray(), 0, 4, -1)

            koblas.axpyVector(2.0, source, destination)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `sparse mutations retain support without scanning large shapes`() {
        val source: Vector = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, Int.MAX_VALUE - 1), doubleArrayOf(0.0, 2.0))
        val destination = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(0, Int.MAX_VALUE - 1), doubleArrayOf(3.0, -4.0))
        val expectedValues = doubleArrayOf(3.0, 0.0, -4.0)
        KoblasEngine.scalar.kernels.axpy(intArrayOf(1, 2), 0, doubleArrayOf(0.0, 2.0), 0, 2, 2.0, expectedValues)
        KoblasEngine.scalar.kernels.scale(expectedValues, 0, -2.0, expectedValues.size)

        koblas.axpyVector(2.0, source, destination)
        koblas.scaleVector(-2.0, destination)

        assertContentEquals(intArrayOf(0, 1, Int.MAX_VALUE - 1), destination.indices)
        assertAxpyAgreesWithReference(expectedValues, destination.values)
    }

    @Test
    fun `sparse axpy preserves entries across buffer growth boundaries`() {
        for (count in intArrayOf(0, 15, 16, 17, 31, 32, 33, 511)) {
            for (sparseSource in booleanArrayOf(false, true)) {
                val values = DoubleArray(count) { if (it % 7 == 0) 0.0 else it * 0.125 }
                val source: Vector = if (sparseSource) {
                    SparseVector.wrap(count, IntArray(count) { it }, values)
                } else {
                    DenseVector.wrap(values)
                }
                val destination = SparseVector.wrap(count, IntArray(count) { it }, DoubleArray(count) { -1.0 })
                val expected = destination.values.copyOf()
                KoblasEngine.scalar.kernels.axpy(expected, 0, 0.5, values, 0, count)

                koblas.axpyVector(0.5, source, destination)

                assertContentEquals(IntArray(count) { it }, destination.copyIndices())
                assertAxpyAgreesWithReference(expected, destination.values)
            }
        }
    }

    @Test
    fun `sparse axpy snapshots a custom source borrowing destination values`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val source: Vector = ForeignVector(DenseVector.wrap(backing, 3, 4, -1))
        val destination = SparseVector.wrap(4, intArrayOf(0, 1, 2, 3), backing)
        val expected = backing.copyOf()
        KoblasEngine.scalar.kernels.axpy(expected, 0, 2.0, source.toDoubleArray(), 0, 4)

        koblas.axpyVector(2.0, source, destination)

        assertSame(backing, destination.values)
        assertAxpyAgreesWithReference(expected, backing)
    }

    @Test
    fun `zero axpy validates every destination without reading operands`() {
        val source = object : Vector {
            override val size = 4
            override fun get(i: Int): Double = error("Unexpected read")
            override fun set(i: Int, v: Double) = error("Unexpected write")
            override fun toDoubleArray(): DoubleArray = error("Snapshotted opaque storage")
        }
        for (destination in mutationOperands(doubleArrayOf(1.0, 0.0, 2.0, 3.0))) {
            val expected = destination.toDoubleArray()

            koblas.axpyVector(0.0, source, destination)

            assertContentEquals(expected, destination.toDoubleArray())
            assertFailsWith<DimensionMismatch> { koblas.axpyVector(0.0, DenseVector.zero(3), destination) }
        }
    }

    private fun mutationOperands(values: DoubleArray): List<Vector> = buildList {
        add(DenseVector.of(values))
        for (stride in intArrayOf(-3, -1, 1, 3)) {
            val backing = DoubleArray(values.size * 3 + 4) { Double.NaN }
            val offset = if (stride > 0 || values.isEmpty()) 2 else 2 + (values.size - 1) * -stride
            val view = DenseVector.wrap(backing, offset, values.size, stride)
            for (i in values.indices) view[i] = values[i]
            add(view)
        }
        val indices = values.indices.filter { values[it] != 0.0 || it % 2 == 0 }.toIntArray()
        add(SparseVector.of(values.size, indices, DoubleArray(indices.size) { values[indices[it]] }))
        add(ForeignVector(DenseVector.of(values)))
    }

    @Test
    fun `sparse norms use selected kernels over stored values only`() {
        val source = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, 100), doubleArrayOf(2.0, -3.0))
        val calls = mutableListOf<String>()
        val engine = object : KernelProvider {
            override val name = "recording"
            override fun asum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "asum"
                return KoblasEngine.scalar.kernels.asum(v, vOff, len, vStride)
            }
            override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "nrm2"
                return KoblasEngine.scalar.kernels.nrm2(v, vOff, len, vStride)
            }
            override fun iamax(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Int {
                assertSame(source.values, v)
                assertEquals(2, len)
                calls += "iamax"
                return KoblasEngine.scalar.kernels.iamax(v, vOff, len, vStride)
            }
        }

        for (type in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) engine.normVector(source, type)

        assertEquals(listOf("asum", "nrm2", "iamax", "iamax"), calls)
    }

    @Test
    fun `sparse sum uses the selected kernel over stored values only`() {
        val source = SparseVector.wrap(Int.MAX_VALUE, intArrayOf(1, 100), doubleArrayOf(2.0, -3.0))
        var length = -1
        val engine = object : KernelProvider {
            override val name = "recording"
            override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(source.values, v)
                length = len
                return KoblasEngine.scalar.kernels.sum(v, vOff, len, vStride)
            }
        }
        val expected = KoblasEngine.scalar.kernels.sum(source.values, 0, source.values.size)

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
        val expected = KoblasEngine.scalar.kernels.dot(source.values, 0, doubleArrayOf(4.0, -2.0), 0, 2)

        for ((x, y) in listOf(source to other, other to source)) {
            reads.clear()

            val actual = KoblasEngine.scalar.dotVector(x, y)

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
                val destination = DenseVector.wrap(backing, if (stride < 0) 11 else 2, 4, stride)
                val expected = backing.copyOf()
                val packed = destination.toDoubleArray()
                if (copy) {
                    packed.fill(0.0)
                    KoblasEngine.scalar.kernels.scatter(source.indices, 0, source.values, 0, 2, packed)
                } else {
                    KoblasEngine.scalar.kernels.axpy(source.indices, 0, source.values, 0, 2, -1.5, packed)
                }
                for (i in packed.indices) expected[destination.offset + i * stride] = packed[i]

                if (copy) {
                    koblas.copyIntoVector(source, destination)
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
                KoblasEngine.scalar.kernels.axpy(
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
        val source = DenseVector.wrap(backing, 2, 5)
        val destination = DenseVector.wrap(backing, 6, 5, -1)
        val expected = backing.copyOf()
        KoblasEngine.scalar.kernels.axpy(expected, 6, 0.75, source.toDoubleArray(), 0, 5, -1)

        koblas.axpyVector(0.75, source, destination)

        assertAxpyAgreesWithReference(expected, backing)
    }

    @Test
    fun `sparse axpy accepts a prefix view and preserves its padding`() {
        for (size in intArrayOf(0, 3)) {
            val backing = DoubleArray(7) { it + 0.5 }
            val destination = DenseVector.wrap(backing, 0, size)
            val source = if (size == 0) sparse(0) else sparse(size, 0 to 2.0, 2 to -1.0)
            val expected = backing.copyOf()
            KoblasEngine.scalar.kernels.axpy(source.indices, 0, source.values, 0, source.indices.size, 2.0, expected)

            koblas.axpyVector(2.0, source, destination)

            assertAxpyAgreesWithReference(expected, backing)
        }
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
            val destination = if (borrowed) DenseVector.wrap(backing, 1, 4) else DenseVector.wrap(backing)
            val source = ForeignVector(DenseVector.wrap(backing, destination.offset + 3, 4, -1))
            val expected = backing.copyOf()
            KoblasEngine.scalar.kernels.axpy(expected, destination.offset, 2.0, source.toDoubleArray(), 0, 4)

            koblas.axpyVector(2.0, source, destination)

            assertAxpyAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `borrowed axpy with sparse input leaves unstored positions untouched`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0)
        val source = sparse(3, 1 to 2.0)
        val destination = DenseVector.wrap(backing, 2, 3, -1)
        val expected = DenseVector.of(destination.toDoubleArray())
        koblas.axpyVector(Double.POSITIVE_INFINITY, source, expected)

        koblas.axpyVector(Double.POSITIVE_INFINITY, source, destination)

        assertContentEquals(expected.values.reversedArray(), backing)
    }

    @Test
    fun `scaling a borrowed slice leaves other entries untouched`() {
        val backing = DoubleArray(8) { it + 1.0 }
        val even = DenseVector.wrap(backing, 0, 4, 2)
        val expected = backing.copyOf()
        ScalarEngine.scale(expected, 0, 2.0, 4, 2)

        koblas.scaleVector(2.0, even)

        assertContentEquals(expected, backing)
    }

    private fun assertWrapperAgreesWithReference(expected: Double, actual: Double) =
        assertClose(expected, actual, "public wrapper reduction")

    private class ForeignVector(private val backing: DenseVector) : Vector {
        override val size: Int get() = backing.size
        override fun get(i: Int): Double = backing[i]
        override fun set(i: Int, v: Double) {
            backing[i] = v
        }
    }

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
                    val source = DenseVector.wrap(backing, offset, size, stride)
                    add(source to if (sameWrapper) source else source.view(0, source.size))
                }
            }
        }
    }
}
