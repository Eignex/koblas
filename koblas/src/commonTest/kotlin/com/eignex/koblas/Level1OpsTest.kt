package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class Level1OpsTest {

    private val sparse = SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    @Test
    fun `norm2 matches the hand value on dense and sparse`() {
        assertEquals(5.0, norm2(DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0))))
        assertEquals(sqrt(13.0), norm2(sparse), 1e-15)
        assertEquals(0.0, norm2(DenseVector.zero(4)))
        assertEquals(0.0, norm2(DenseVector.zero(0)))
    }

    @Test
    fun `norm2 survives overflow and underflow via the rescale fallback`() {
        // Squares of 1e200 overflow a plain sum; the rescale recovers the exact 3-4-5 triangle.
        assertEquals(5.0e200, norm2(DenseVector.of(doubleArrayOf(3.0e200, 0.0, -4.0e200))), 1e186)
        // Squares of 1e-200 underflow to zero; the rescale recovers them.
        assertEquals(5.0e-200, norm2(DenseVector.of(doubleArrayOf(3.0e-200, 4.0e-200))), 1e-214)
        assertEquals(1.0e-300, norm2(DenseVector.of(doubleArrayOf(1.0e-300))), 1e-314)
        // Sparse vectors take the same fallback.
        val sparseHuge = SparseVector.of(5, intArrayOf(0, 3), doubleArrayOf(3.0e200, 4.0e200))
        assertEquals(5.0e200, norm2(sparseHuge), 1e186)
        // Non-finite inputs propagate: NaN stays NaN, an infinite component gives an infinite norm.
        assertTrue(norm2(DenseVector.of(doubleArrayOf(1.0, Double.NaN))).isNaN())
        assertTrue(norm2(DenseVector.of(doubleArrayOf(1.0e200, Double.NaN))).isNaN())
        assertEquals(Double.POSITIVE_INFINITY, norm2(DenseVector.of(doubleArrayOf(1.0, Double.NEGATIVE_INFINITY))))
    }

    @Test
    fun `asum matches the hand value on dense and sparse`() {
        assertEquals(7.0, asum(DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0))))
        assertEquals(5.0, asum(sparse))
        assertEquals(0.0, asum(DenseVector.zero(0)))
    }

    @Test
    fun `iamax returns the first maximal index and handles edge cases`() {
        assertEquals(2, iamax(DenseVector.of(doubleArrayOf(1.0, -2.0, 5.0, -5.0))))
        assertEquals(1, iamax(DenseVector.of(doubleArrayOf(1.0, -5.0, 5.0)))) // tie: first wins
        assertEquals(4, iamax(sparse))
        assertEquals(0, iamax(DenseVector.zero(3))) // zero vector: first element
        assertEquals(0, iamax(SparseVector.of(3, IntArray(0), DoubleArray(0)))) // all-unstored: same
        assertEquals(-1, iamax(DenseVector.zero(0)))
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
            assertTrue(abs(norm2(v) - sqrt(sumSq)) <= 1e-12 * sqrt(sumSq))
            assertTrue(abs(asum(v) - sumAbs) <= 1e-12 * sumAbs)
            assertEquals(maxIdx, iamax(v))
        }
    }

    @Test
    fun `sparse and dense carriers of the same vector agree on every op`() {
        assertEquals(norm2(denseOfSparse), norm2(sparse), 1e-15)
        assertEquals(asum(denseOfSparse), asum(sparse), 1e-15)
        assertEquals(iamax(denseOfSparse), iamax(sparse))
    }
}
