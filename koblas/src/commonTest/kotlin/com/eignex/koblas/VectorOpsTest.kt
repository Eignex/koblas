package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class VectorOpsTest {
    private val sparse = SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    private fun dense(vararg values: Double) = DenseVector.of(values)
    private fun sparse(size: Int, vararg entries: Pair<Int, Double>) = SparseVector.of(
        size,
        entries.map { it.first }.toIntArray(),
        entries.map { it.second }.toDoubleArray(),
    )

    @Test
    fun `dot is symmetric and sparsity-agnostic`() {
        val a = dense(1.0, 2.0, 3.0, 4.0)
        val b = dense(0.5, 0.0, -1.0, 2.0)
        val bSparse = sparse(4, 0 to 0.5, 2 to -1.0, 3 to 2.0)
        val expected = 1 * 0.5 + 2 * 0 + 3 * (-1) + 4 * 2
        assertEquals(expected, a dot b, 1e-12)
        assertEquals(expected, b dot a, 1e-12)
        assertEquals(expected, a dot bSparse, 1e-12)
        assertEquals(expected, bSparse dot a, 1e-12)
        assertEquals(0.5 * 0.5 + 0 + 1 + 4, bSparse dot b, 1e-12)
    }

    @Test
    fun `axpy adds alpha-scaled x to y for any sparsity`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        y.axpy(2.0, sparse(3, 0 to 1.0, 2 to -1.0))
        assertEquals(dense(3.0, 2.0, 1.0), y)
    }

    @Test
    fun `scale mutates in place and respects identity`() {
        val v = DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        v.scale(0.5)
        assertEquals(dense(0.5, -1.0, 1.5), v)
        v.scale(1.0) // no-op
        assertEquals(dense(0.5, -1.0, 1.5), v)
    }

    @Test
    fun `norm2 matches the hand value on dense and sparse`() {
        assertEquals(5.0, DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0)).norm2())
        assertEquals(sqrt(13.0), sparse.norm2(), 1e-15)
        assertEquals(0.0, DenseVector.zero(4).norm2())
        assertEquals(0.0, DenseVector.zero(0).norm2())
    }

    @Test
    fun `norm2 survives overflow and underflow via the rescale fallback`() {
        assertEquals(5.0e200, DenseVector.of(doubleArrayOf(3.0e200, 0.0, -4.0e200)).norm2(), 1e186)
        assertEquals(5.0e-200, DenseVector.of(doubleArrayOf(3.0e-200, 4.0e-200)).norm2(), 1e-214)
        assertEquals(1.0e-300, DenseVector.of(doubleArrayOf(1.0e-300)).norm2(), 1e-314)
        val sparseHuge = SparseVector.of(5, intArrayOf(0, 3), doubleArrayOf(3.0e200, 4.0e200))
        assertEquals(5.0e200, sparseHuge.norm2(), 1e186)
        assertTrue(DenseVector.of(doubleArrayOf(1.0, Double.NaN)).norm2().isNaN())
        assertTrue(DenseVector.of(doubleArrayOf(1.0e200, Double.NaN)).norm2().isNaN())
        assertEquals(Double.POSITIVE_INFINITY, DenseVector.of(doubleArrayOf(1.0, Double.NEGATIVE_INFINITY)).norm2())
    }

    @Test
    fun `sum adds the entries for any storage`() {
        val values = doubleArrayOf(2.0, -3.5, 0.0, 4.25, -1.0, 0.5)
        val expected = values.sum()
        val strided = StridedVectorView(DoubleArray(12) { values[it / 2] }, 0, 6, 2)
        assertEquals(expected, DenseVector.of(values).sum(), 1e-12, "dense")
        assertEquals(expected, strided.sum(), 1e-12, "strided")
        assertEquals(-1.0, sparse.sum(), 1e-12, "sparse stores 2 and -3")
        assertEquals(0.0, DenseVector.of(DoubleArray(0)).sum(), "empty")
    }

    @Test
    fun `sum and asum differ once a sign does`() {
        val mixed = dense(3.0, -4.0)
        assertEquals(-1.0, mixed.sum(), 1e-12, "sum")
        assertEquals(7.0, mixed.asum(), 1e-12, "asum")
    }

    @Test
    fun `compensatedSum recovers what a naive sum loses`() {
        // The exact total is 2. A naive left-to-right sum drops both ones against the large terms and
        // answers 0, whichever order the large terms come in.
        val values = doubleArrayOf(1.0, 1e100, 1.0, -1e100)
        assertEquals(0.0, values.sum(), "the naive sum is the thing being fixed")
        assertEquals(2.0, DenseVector.of(values).compensatedSum(), "dense")
        val strided = StridedVectorView(DoubleArray(8) { values[it / 2] }, 0, 4, 2)
        assertEquals(2.0, strided.compensatedSum(), "strided takes the same path as any generic storage")
    }

    @Test
    fun `compensatedSum agrees with sum on benign input`() {
        val rng = Random(20260903)
        val v = DenseVector.of(randomVector(500, rng))
        assertEquals(v.sum(), v.compensatedSum(), 1e-9, "no drift on well-scaled input")
    }

    @Test
    fun `asum matches the hand value on dense and sparse`() {
        assertEquals(7.0, DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0)).asum())
        assertEquals(5.0, sparse.asum())
        assertEquals(0.0, DenseVector.zero(0).asum())
    }

    @Test
    fun `iamax returns the first maximal index and handles edge cases`() {
        assertEquals(2, DenseVector.of(doubleArrayOf(1.0, -2.0, 5.0, -5.0)).iamax())
        assertEquals(1, DenseVector.of(doubleArrayOf(1.0, -5.0, 5.0)).iamax()) // tie: first wins
        assertEquals(4, sparse.iamax())
        assertEquals(0, DenseVector.zero(3).iamax()) // zero vector: first element
        assertEquals(0, SparseVector.of(3, IntArray(0), DoubleArray(0)).iamax()) // all-unstored: same
        assertEquals(-1, DenseVector.zero(0).iamax())
    }

    @Test
    fun `iamax reads a stored zero as the zero it is`() {
        assertEquals(0, SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(0, SparseVector.of(5, intArrayOf(1, 3), doubleArrayOf(0.0, -0.0)).iamax())
        assertEquals(DenseVector.zero(5).iamax(), SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(4, SparseVector.of(5, intArrayOf(1, 4), doubleArrayOf(0.0, -2.0)).iamax())
    }

    @Test
    fun `iamax preserves nan and logical index semantics`() {
        assertEquals(2, DenseVector.of(doubleArrayOf(Double.NaN, -2.0, Double.POSITIVE_INFINITY)).iamax())
        assertEquals(0, DenseVector.of(doubleArrayOf(Double.NaN, Double.NaN)).iamax())
        assertEquals(0, SparseVector.of(4, intArrayOf(2), doubleArrayOf(Double.NaN)).iamax())
        val backing = doubleArrayOf(99.0, Double.NaN, 99.0, -7.0, 99.0, 6.0)
        assertEquals(1, StridedVectorView(backing, offset = 1, size = 3, stride = 2).iamax())
    }

    @Test
    fun `copy replicates dense and sparse sources and rejects size mismatch`() {
        val dst = DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0, 9.0, 9.0, 9.0))
        copy(sparse, dst) // sparse: must zero-fill the unstored slots
        assertContentEquals(denseOfSparse.data, dst.data)
        copy(DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), dst)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dst.data)
        assertFailsWith<DimensionMismatch> { copy(DenseVector.zero(2), DenseVector.zero(3)) }
    }

    @Test
    fun `swap exchanges contents and rejects size mismatch`() {
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(3.0, 4.0))
        swap(a, b)
        assertContentEquals(doubleArrayOf(3.0, 4.0), a.data)
        assertContentEquals(doubleArrayOf(1.0, 2.0), b.data)
        assertFailsWith<DimensionMismatch> { swap(DenseVector.zero(2), DenseVector.zero(3)) }
    }

    @Test
    fun `swap snapshots overlapping borrowed slices`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val a = StridedVectorView(backing, 0, 4)
        val b = StridedVectorView(backing, 4, 4, -1)
        val originalA = a.toDoubleArray()
        val originalB = b.toDoubleArray()
        val expected = backing.copyOf()
        for (i in 0 until a.size) expected[a.offset + i * a.stride] = originalB[i]
        for (i in 0 until b.size) expected[b.offset + i * b.stride] = originalA[i]

        swap(a, b)

        assertContentEquals(expected, backing)
    }

    @Test
    fun `vector operations reject mismatched sizes`() {
        assertFailsWith<DimensionMismatch> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<DimensionMismatch> { dense(1.0).axpy(1.0, dense(1.0, 2.0)) }
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
    fun `mixed sparse and dense dot agrees in both operand orders`() {
        val sparse = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val dense = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = 2.0 * 2.0 + -3.0 * 5.0
        assertEquals(expected, sparse dot dense)
        assertEquals(expected, dense dot sparse)
    }

    @Test
    fun `sparse dot ignores unstored entries in borrowed and foreign vectors`() {
        val sparse = sparse(3, 1 to 2.0)
        val backing = doubleArrayOf(Double.NaN, 3.0, Double.POSITIVE_INFINITY)
        val borrowed = StridedVectorView(backing, 2, 3, -1)
        val foreign = object : Vector {
            override val size: Int = backing.size
            override fun get(i: Int): Double = backing[i]
            override fun toDoubleArray(): DoubleArray = backing.copyOf()
        }

        for (other in listOf(borrowed, foreign)) {
            val expected = sparse dot DenseVector.of(other.toDoubleArray())

            assertEquals(expected, sparse dot other)
            assertEquals(expected, other dot sparse)
        }
    }

    @Test
    fun `strided norms preserve infinity and nan semantics`() {
        val cases = listOf(
            doubleArrayOf(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY),
            doubleArrayOf(Double.POSITIVE_INFINITY, 3.0, Double.POSITIVE_INFINITY),
            doubleArrayOf(Double.POSITIVE_INFINITY, Double.NaN),
            doubleArrayOf(Double.NaN, Double.POSITIVE_INFINITY),
        )
        for (values in cases) {
            for (stride in intArrayOf(2, -2)) {
                val backing = DoubleArray(values.size * 2) { Double.NaN }
                val view = StridedVectorView(backing, if (stride > 0) 0 else backing.size - 2, values.size, stride)
                for (i in values.indices) view[i] = values[i]
                val expected = com.eignex.koblas.dense.ScalarVectorKernels.nrm2(values, 0, values.size)

                val actual = view.norm2()

                assertEquals(expected, actual, "values=${values.toList()} stride=$stride")
            }
        }
    }

    @Test
    fun `copy into dense storage preserves a borrowed source sharing its buffer`() {
        for (reverse in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = StridedVectorView(backing, if (reverse) 3 else 0, 4, if (reverse) -1 else 1)
            val expected = source.toDoubleArray()

            copy(source, DenseVector.wrap(backing))

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `copy between overlapping borrowed slices preserves the input sequence`() {
        for (stride in intArrayOf(1, -1)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
            val source = StridedVectorView(backing, if (stride > 0) 0 else 4, 4, stride)
            val destination = StridedVectorView(backing, if (stride > 0) 1 else 3, 4, stride)
            val expected = backing.copyOf()
            val snapshot = source.toDoubleArray()
            for (i in snapshot.indices) expected[destination.offset + i * stride] = snapshot[i]

            copy(source, destination)

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
                copy(source, StridedVectorView(backing, 2, 3, -1))
            } else {
                copy(source, DenseVector.wrap(backing))
            }

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `axpy preserves a borrowed source sharing the destination buffer`() {
        for (borrowed in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            val source = StridedVectorView(backing, 3, 4, -1)
            val expected = DoubleArray(backing.size) { backing[it] + 2.0 * source[it] }

            if (borrowed) {
                StridedVectorView(backing, 0, 4).axpy(2.0, source)
            } else {
                DenseVector.wrap(backing).axpy(2.0, source)
            }

            assertContentEquals(expected, backing)
        }
    }

    @Test
    fun `borrowed axpy with sparse input leaves unstored positions untouched`() {
        val backing = doubleArrayOf(1.0, 2.0, 3.0)
        val source = sparse(3, 1 to 2.0)
        val destination = StridedVectorView(backing, 2, 3, -1)
        val expected = DenseVector.of(destination.toDoubleArray())
        expected.axpy(Double.POSITIVE_INFINITY, source)

        destination.axpy(Double.POSITIVE_INFINITY, source)

        assertContentEquals(expected.data.reversedArray(), backing)
    }

    @Test
    fun `level-1 ops agree with naive references on random vectors`() {
        val rng = Random(20260727)
        repeat(20) {
            val n = rng.nextInt(1, 200)
            val data = DoubleArray(n) { rng.nextDouble(-100.0, 100.0) }
            val v = DenseVector.of(data)
            var sumSq = 0.0
            var sumAbs = 0.0
            var maxIdx = 0
            for (i in 0 until n) {
                sumSq += data[i] * data[i]
                sumAbs += abs(data[i])
                if (abs(data[i]) > abs(data[maxIdx])) maxIdx = i
            }
            assertTrue(abs(v.norm2() - sqrt(sumSq)) <= 1e-12 * sqrt(sumSq))
            assertTrue(abs(v.asum() - sumAbs) <= 1e-12 * sumAbs)
            assertEquals(maxIdx, v.iamax())
        }
    }
}
