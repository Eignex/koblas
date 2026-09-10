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
    fun `norm1 is the maximum absolute column sum`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, 5.0, -6.0),
            ),
        )
        assertEquals(9.0, a.norm1()) // columns sum to 5, 7, 9
        assertEquals(0.0, DenseMatrix(0, 3).norm1())
        assertEquals(0.0, DenseMatrix(3, 0).norm1())
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
        assertFailsWith<IllegalArgumentException> { copy(DenseVector.zero(2), DenseVector.zero(3)) }
    }

    @Test
    fun `swap exchanges contents and rejects size mismatch`() {
        val a = DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = DenseVector.of(doubleArrayOf(3.0, 4.0))
        swap(a, b)
        assertContentEquals(doubleArrayOf(3.0, 4.0), a.data)
        assertContentEquals(doubleArrayOf(1.0, 2.0), b.data)
        assertFailsWith<IllegalArgumentException> { swap(DenseVector.zero(2), DenseVector.zero(3)) }
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
