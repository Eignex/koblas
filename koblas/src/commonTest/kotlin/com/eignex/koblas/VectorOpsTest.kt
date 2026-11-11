package com.eignex.koblas

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.operations.dotVector
import kotlin.math.abs
import kotlin.test.*

class VectorOpsTest {
    @Test
    fun `sparse copies preserve large shapes and stored zeros without densifying`() {
        val source: Vector = SparseVector.wrap(
            Int.MAX_VALUE,
            intArrayOf(0, Int.MAX_VALUE - 1),
            doubleArrayOf(0.0, -2.0),
        )
        val destination: Vector = SparseVector.wrap(Int.MAX_VALUE, IntArray(0), DoubleArray(0))

        val result = source.copy()
        destination.copyFrom(source)

        assertEquals(source, result)
        assertEquals(source, destination)
        assertNotSame(assertIs<SparseVector>(source).values, assertIs<SparseVector>(result).values)
        assertNotSame(source.values, assertIs<SparseVector>(destination).values)
    }

    @Test
    fun `copies own their storage across vector implementations`() {
        val values = doubleArrayOf(1.5, 0.0, -0.0, -2.0)
        for (source in normOperands(values)) {
            val result = source.copy()

            assertContentEquals(values, result.toDoubleArray())
            if (source is SparseVector) {
                val sparseCopy = assertIs<SparseVector>(result)
                assertContentEquals(source.indices, sparseCopy.indices)
                assertNotSame(source.indices, sparseCopy.indices)
                assertNotSame(source.values, sparseCopy.values)
            }
            result[0] = 9.0
            assertContentEquals(values, source.toDoubleArray())
        }
    }

    @Test
    fun `copy from supports every source and destination storage`() {
        val values = doubleArrayOf(1.5, 0.0, -0.0, -2.0)
        for (source in normOperands(values)) {
            for (destination in normOperands(DoubleArray(values.size) { 7.0 })) {
                destination.copyFrom(source)

                assertContentEquals(values, destination.toDoubleArray())
                if (destination is SparseVector && source is SparseVector) {
                    assertContentEquals(source.indices, destination.indices)
                    assertNotSame(source.values, destination.values)
                }
            }
        }
    }

    @Test
    fun `copy from snapshots overlapping custom destinations`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val source: Vector = ForeignVector(DenseVector.wrap(backing, 3, 4, -1))
        val destination: Vector = ForeignVector(DenseVector.wrap(backing))
        val expected = source.toDoubleArray()

        destination.copyFrom(source)

        assertContentEquals(expected, backing)
    }

    @Test
    fun `identical mapping copy preserves the full backing buffer`() {
        for ((source, destination) in identicalDenseMappings()) {
            val expected = source.values.copyOf()

            destination.copyFrom(source)

            assertContentEquals(expected, source.values)
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

    @Test
    fun `sparse dot against dense views agrees with the scalar reference`() {
        val source = sparse(5, 0 to 2.0, 2 to -1.5, 4 to 0.0)
        for (stride in intArrayOf(-3, -1, 1, 3)) {
            val backing = DoubleArray(18) { it * 0.125 - 0.75 }
            val view = DenseVector.wrap(backing, if (stride > 0) 2 else 14, 5, stride)
            val expected = KoblasEngine.scalar.dotVector(source, DenseVector.wrap(view.toDoubleArray()))

            assertDotAgreesWithReference(expected, source dot view)
            assertDotAgreesWithReference(expected, view dot source)
        }
    }

    @Test
    fun `sparse dot against dense views evaluates stored zeros`() {
        val source = sparse(3, 1 to 0.0)
        val view = DenseVector.wrap(doubleArrayOf(Double.NaN, 1.0, Double.POSITIVE_INFINITY, 1.0, Double.NaN), 0, 3, 2)
        val expected = KoblasEngine.scalar.dotVector(source, DenseVector.wrap(view.toDoubleArray()))

        assertDotAgreesWithReference(expected, source dot view)
        assertDotAgreesWithReference(expected, view dot source)
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double) {
        if (expected.isNaN()) assertTrue(actual.isNaN()) else assertEquals(expected, actual, 1e-12)
    }

    /** A non-koblas vector storage, which reaches the public adapter fallbacks rather than a kernel. */
    private class ForeignVector(private val backing: DenseVector) : Vector {
        override val size: Int get() = backing.size
        override fun get(i: Int): Double = backing[i]
        override fun set(i: Int, v: Double) {
            backing[i] = v
        }
    }

    private fun foreignRamp(size: Int) = ForeignVector(DenseVector.of(DoubleArray(size) { it * 0.5 - 1.0 }))

    private val sparse = SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    private fun dense(vararg values: Double) = DenseVector.of(values)
    private fun sparse(size: Int, vararg entries: Pair<Int, Double>) = SparseVector.of(
        size,
        entries.map { it.first }.toIntArray(),
        entries.map { it.second }.toDoubleArray(),
    )

    @Test
    fun `dot is symmetric and sparsity agnostic`() {
        val a = dense(1.0, 2.0, 3.0, 4.0)
        val b = dense(0.5, 0.0, -1.0, 2.0)
        val bSparse = sparse(4, 0 to 0.5, 2 to -1.0, 3 to 2.0)
        val expected = KoblasEngine.scalar.kernels.dot(a.values, 0, b.values, 0, 4)
        val sparseByDense = KoblasEngine.scalar.kernels.dot(b.values, 0, b.values, 0, 4)

        for ((actual, context) in listOf(
            (a dot b) to "dense by dense",
            (b dot a) to "dense by dense reversed",
            (a dot bSparse) to "dense by sparse",
            (bSparse dot a) to "sparse by dense",
        )) {
            assertReductionAgreesWithReference(expected, actual, context)
        }
        assertReductionAgreesWithReference(sparseByDense, bSparse dot b, "sparse by its own dense form")
    }

    @Test
    fun `supported norms agree with the scalar oracle across vector storage`() {
        val values = doubleArrayOf(-1.5, 0.0, 3.5, -7.25, 0.0, 2.0)
        for (source in normOperands(values)) {
            for (type in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) {
                val expected = referenceNorm(values, type)

                val actual = source.norm(type)

                assertReductionAgreesWithReference(expected, actual, "norm type $type")
            }
        }
    }

    @Test
    fun `euclidean norm rescales across vector storage`() {
        for (scale in doubleArrayOf(1e-200, 1e200)) {
            val values = doubleArrayOf(3.0 * scale, 0.0, -4.0 * scale)
            for (source in normOperands(values)) {
                val expected = referenceNorm(values, Norm.Two)

                val actual = source.norm(Norm.Two)

                assertReductionAgreesWithReference(expected / scale, actual / scale, "rescaled norm")
            }
        }
    }

    @Test
    fun `supported norms of empty and zero vectors are zero`() {
        val emptySparse = SparseVector.wrap(Int.MAX_VALUE, IntArray(0), DoubleArray(0))
        val emptyView = DenseVector.wrap(doubleArrayOf(Double.NaN), 1, 0, -3)
        for (source in normOperands(DoubleArray(0)) + normOperands(DoubleArray(4)) + listOf(emptySparse, emptyView)) {
            for (type in listOf(Norm.One, Norm.Two, Norm.Infinity, Norm.MaxAbs)) {
                assertEquals(0.0, source.norm(type))
            }
        }
    }

    @Test
    fun `unsupported vector norm is rejected before reading the vector`() {
        val source = object : Vector {
            override val size: Int get() = error("Read vector size")
            override fun set(i: Int, v: Double) = error("Unexpected mutation")
            override fun get(i: Int): Double = error("Read vector entry")
            override fun toDoubleArray(): DoubleArray = error("Materialized vector")
        }

        assertFailsWith<IllegalArgumentException> { source.norm(Norm.Frobenius) }
    }

    private fun normOperands(values: DoubleArray): List<Vector> = buildList {
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

    private fun referenceNorm(values: DoubleArray, type: Norm): Double = when (type) {
        Norm.One -> KoblasEngine.scalar.kernels.asum(values, 0, values.size)

        Norm.Two -> KoblasEngine.scalar.kernels.nrm2(values, 0, values.size)

        Norm.Infinity, Norm.MaxAbs -> {
            val index = KoblasEngine.scalar.kernels.iamax(values, 0, values.size)
            if (index < 0) 0.0 else abs(values[index])
        }

        Norm.Frobenius -> error("Unsupported vector norm")
    }

    @Test
    fun `norm two survives overflow and underflow via the rescale fallback`() {
        assertEquals(5.0e200, DenseVector.of(doubleArrayOf(3.0e200, 0.0, -4.0e200)).norm(), 1e186)
        assertEquals(5.0e-200, DenseVector.of(doubleArrayOf(3.0e-200, 4.0e-200)).norm(), 1e-214)
        assertEquals(1.0e-300, DenseVector.of(doubleArrayOf(1.0e-300)).norm(), 1e-314)
        val sparseHuge = SparseVector.of(5, intArrayOf(0, 3), doubleArrayOf(3.0e200, 4.0e200))
        assertEquals(5.0e200, sparseHuge.norm(), 1e186)
    }

    @Test
    fun `sum adds the entries for any storage`() {
        val values = doubleArrayOf(2.0, -3.5, 0.0, 4.25, -1.0, 0.5)
        val expected = KoblasEngine.scalar.kernels.sum(values, 0, values.size)
        val strided = DenseVector.wrap(DoubleArray(12) { values[it / 2] }, 0, 6, 2)
        assertReductionAgreesWithReference(expected, DenseVector.of(values).sum(), "dense")
        assertReductionAgreesWithReference(expected, strided.sum(), "strided")
        assertReductionAgreesWithReference(
            KoblasEngine.scalar.kernels.sum(denseOfSparse.values, 0, denseOfSparse.size),
            sparse.sum(),
            "sparse",
        )
        assertEquals(0.0, DenseVector.of(DoubleArray(0)).sum(), "empty")
    }

    @Test
    fun `asum agrees with the scalar oracle on dense and sparse`() {
        val values = doubleArrayOf(3.0, 0.0, -4.0)
        assertReductionAgreesWithReference(
            KoblasEngine.scalar.kernels.asum(values, 0, values.size),
            DenseVector.of(values).asum(),
            "dense",
        )
        assertReductionAgreesWithReference(
            KoblasEngine.scalar.kernels.asum(denseOfSparse.values, 0, denseOfSparse.size),
            sparse.asum(),
            "sparse",
        )
        assertEquals(0.0, DenseVector.zero(0).asum())
    }

    @Test
    fun `copy replicates dense and sparse sources and rejects size mismatch`() {
        val dst = DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0, 9.0, 9.0, 9.0))
        dst.copyFrom(sparse) // sparse: must zero-fill the unstored slots
        assertContentEquals(denseOfSparse.values, dst.values)
        dst.copyFrom(DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dst.values)
        assertFailsWith<DimensionMismatch> { DenseVector.zero(3).copyFrom(DenseVector.zero(2)) }
    }

    @Test
    fun `sparse against sparse dot matches the dense answer over merge shapes`() {
        val n = 8
        val patterns = listOf(
            intArrayOf(0, 2, 4, 6) to intArrayOf(1, 3, 5, 7),
            intArrayOf(0, 1, 2) to intArrayOf(0, 1, 2),
            intArrayOf(0, 7) to intArrayOf(3, 4),
            intArrayOf(0, 1, 2, 3) to intArrayOf(3),
            intArrayOf(5) to intArrayOf(0, 1, 2, 3, 4),
            IntArray(0) to intArrayOf(0, 4),
            IntArray(0) to IntArray(0),
        )
        for ((ia, ib) in patterns) {
            val a = SparseVector.of(n, ia, DoubleArray(ia.size) { it + 1.5 })
            val b = SparseVector.of(n, ib, DoubleArray(ib.size) { it + 2.5 })
            val expected = DenseVector.of(a.toDoubleArray()) dot DenseVector.of(b.toDoubleArray())
            assertEquals(expected, a dot b, "pattern ${ia.toList()} vs ${ib.toList()}")
            assertEquals(expected, b dot a, "dot should be symmetric for ${ia.toList()} vs ${ib.toList()}")
        }
    }

    @Test
    fun `sparse dot ignores unstored entries in borrowed and foreign vectors`() {
        val sparse = sparse(3, 1 to 2.0)
        val backing = doubleArrayOf(Double.NaN, 3.0, Double.POSITIVE_INFINITY)
        val borrowed = DenseVector.wrap(backing, 2, 3, -1)
        val foreign = ForeignVector(DenseVector.wrap(backing))

        for (other in listOf(borrowed, foreign)) {
            val expected = sparse dot DenseVector.of(other.toDoubleArray())

            assertEquals(expected, sparse dot other)
            assertEquals(expected, other dot sparse)
        }
    }

    @Test
    fun `copy snapshots a foreign source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = if (borrowed) {
                doubleArrayOf(9.0, 1.0, 2.0, 3.0, 4.0, 9.0)
            } else {
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            }
            val destination = if (borrowed) DenseVector.wrap(backing, 1, 4) else DenseVector.wrap(backing)
            val source = ForeignVector(DenseVector.wrap(backing, destination.offset + 3, 4, -1))
            val expected = backing.copyOf()
            val snapshot = source.toDoubleArray()
            snapshot.copyInto(expected, destination.offset)

            destination.copyFrom(source)

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `copy into dense storage preserves a borrowed source sharing its buffer`() {
        for (reverse in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = DenseVector.wrap(backing, if (reverse) 3 else 0, 4, if (reverse) -1 else 1)
            val expected = source.toDoubleArray()

            DenseVector.wrap(backing).copyFrom(source)

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `copy preserves sparse values shared with the destination`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0)
            val source = SparseVector.wrap(3, intArrayOf(0, 1, 2), backing)
            val expected = if (borrowed) backing.reversedArray() else backing.copyOf()

            if (borrowed) {
                DenseVector.wrap(backing, 2, 3, -1).copyFrom(source)
            } else {
                DenseVector.wrap(backing).copyFrom(source)
            }

            assertContentEquals(expected, backing)
        }
    }

    private fun assertReductionAgreesWithReference(expected: Double, actual: Double, context: String) {
        assertClose(expected, actual, context)
    }

    @Test
    fun `the norm accepts a foreign vector`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val expected = ScalarEngine.nrm2(values, 0, values.size)

        val actual = source.norm()

        assertReductionAgreesWithReference(expected, actual, "foreign norm")
    }

    @Test
    fun `the absolute sum accepts a foreign vector`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val expected = ScalarEngine.asum(values, 0, values.size)

        val actual = source.asum()

        assertReductionAgreesWithReference(expected, actual, "foreign asum")
    }

    @Test
    fun `dot accepts a foreign vector on either side`() {
        val source = foreignRamp(9)
        val values = source.toDoubleArray()
        val dense = DenseVector.of(values)
        val expected = ScalarEngine.dot(values, 0, values, 0, values.size)

        for ((left, right) in listOf(source to source, dense to source, source to dense)) {
            val actual = left dot right

            assertReductionAgreesWithReference(expected, actual, "foreign dot")
        }
    }

    @Test
    fun `dot reads the logical entries of borrowed slices`() {
        val backing = DoubleArray(8) { it + 1.0 }
        val even = DenseVector.wrap(backing, 0, 4, 2)
        val odd = DenseVector.wrap(backing, 1, 4, 2)
        val expected = ScalarEngine.dot(backing, 0, backing, 1, 4, 2, 2)

        val actual = even dot odd

        assertClose(expected, actual, "borrowed dot")
    }
}
