@file:Suppress("PropertyName") // math convention: single-letter matrices (A, M) in tests

package com.eignex.koblas

import com.eignex.koblas.dense.koblas
import com.eignex.koblas.dense.matMul
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The standalone vector and matrix operations: the level-1 kernels, the norms, `gemv`, the rank-1
 * `ger` update, and `transpose`.
 *
 * Every operation that accepts a [VectorView] is checked on both a dense and a sparse carrier of the
 * same vector, because the sparse path is a separate loop that skips unstored slots and is where a
 * dispatch mistake produces a plausible but wrong answer. Hand-computed expectations are used wherever
 * the example is small enough to write down.
 */
@Suppress("VariableNaming") // single-letter matrix/vector names track math conventions
class OpsTest {

    private val sparse = SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    private fun dense(vararg v: Double) = DenseVector.of(v)
    private fun sparse(size: Int, vararg pairs: Pair<Int, Double>) =
        SparseVector.of(size, pairs.map { it.first }.toIntArray(), pairs.map { it.second }.toDoubleArray())

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
        // Same vector twice (sparse x dense form) - exercises the sparse-dispatch path.
        assertEquals(0.5 * 0.5 + 0 + 1 + 4, bSparse dot b, 1e-12)
    }

    @Test
    fun `axpy adds alpha-scaled x to y for any sparsity`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        axpy(y, 2.0, sparse(3, 0 to 1.0, 2 to -1.0))
        assertEquals(dense(3.0, 2.0, 1.0), y)
    }

    @Test
    fun `scale mutates in place and respects identity`() {
        val v = DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        scale(v, 0.5)
        assertEquals(dense(0.5, -1.0, 1.5), v)
        scale(v, 1.0) // no-op
        assertEquals(dense(0.5, -1.0, 1.5), v)
    }

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
    fun `norm1 is the maximum absolute column sum`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, 5.0, -6.0),
            ),
        )
        assertEquals(9.0, norm1(a)) // columns sum to 5, 7, 9
        assertEquals(0.0, norm1(DenseMatrix(0, 3)))
        assertEquals(0.0, norm1(DenseMatrix(3, 0)))
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

    @Test
    fun `the gemv overload computes A x for dense and sparse x`() {
        // A = [[1, 2], [3, 4], [5, 6]]
        val A = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(5.0, 6.0),
            ),
        )
        val xDense = dense(1.0, -1.0)
        val xSparse = sparse(2, 0 to 1.0, 1 to -1.0)
        val expected = dense(1.0 * 1 + 2 * -1, 3.0 * 1 + 4 * -1, 5.0 * 1 + 6 * -1)
        assertEquals(expected, gemv(A, xDense))
        assertEquals(expected, gemv(A, xSparse))
    }

    @Test
    fun `sparse and dense gemv agree on a random example`() {
        val rng = Random(7)
        val n = 8
        val A = randomMatrix(n, n, rng)
        // Build a sparse x that touches half the coords.
        val nz = (0 until n).filter { rng.nextBoolean() }
        val xv = DoubleArray(n)
        for (i in nz) xv[i] = rng.nextDouble(-1.0, 1.0)
        val xSparse = SparseVector.of(n, nz.toIntArray(), nz.map { xv[it] }.toDoubleArray())
        assertClose(gemv(A, DenseVector.of(xv)).data, gemv(A, xSparse).data, "gemv dense vs sparse")
    }

    @Test
    fun `ger updates a matrix with alpha x y_transpose`() {
        // M starts as identity 2x2; add 0.5 * [1,2] * [3,4]^T = 0.5 * [[3,4],[6,8]]
        val M = DenseMatrix.diagonal(2, 1.0)
        ger(0.5, dense(1.0, 2.0), dense(3.0, 4.0), M)
        assertEquals(1.0 + 0.5 * 3, M[0, 0], 1e-12)
        assertEquals(0.5 * 4, M[0, 1], 1e-12)
        assertEquals(0.5 * 6, M[1, 0], 1e-12)
        assertEquals(1.0 + 0.5 * 8, M[1, 1], 1e-12)
    }

    @Test
    fun `ger with sparse operands only touches nonzero rows and cols`() {
        val M = DenseMatrix.diagonal(3, 0.0)
        ger(1.0, sparse(3, 1 to 2.0), sparse(3, 0 to 3.0, 2 to 4.0), M)
        // Only row 1 should be nonzero; cols 0 and 2 in that row get 2*3=6 and 2*4=8.
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                val expected = when {
                    i == 1 && j == 0 -> 6.0
                    i == 1 && j == 2 -> 8.0
                    else -> 0.0
                }
                assertEquals(expected, M[i, j], 1e-12, "M[$i,$j]")
            }
        }
    }

    @Test
    fun `alpha zero makes axpy and ger no-ops`() {
        val y = DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        axpy(y, 0.0, DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0)))
        assertTrue(y.toDoubleArray().contentEquals(doubleArrayOf(1.0, 2.0, 3.0)))
        val M = DenseMatrix.diagonal(2, 1.0)
        ger(0.0, DenseVector.of(doubleArrayOf(1.0, 1.0)), DenseVector.of(doubleArrayOf(1.0, 1.0)), M)
        for (i in 0 until 2) {
            for (j in 0 until 2) assertEquals(if (i == j) 1.0 else 0.0, M[i, j])
        }
    }

    @Test
    fun `ger skips zero entries on both carriers`() {
        // Dense: a zero in x leaves that whole row untouched.
        val dense = DenseMatrix.diagonal(3, 0.0)
        ger(1.0, dense(0.0, 2.0, 0.0), dense(1.0, 1.0, 1.0), dense)
        for (i in 0 until 3) {
            for (j in 0 until 3) assertEquals(if (i == 1) 2.0 else 0.0, dense[i, j], 1e-12, "dense[$i,$j]")
        }
        // Sparse: a stored zero counts as a zero, not as a stored entry to scatter.
        val sparse = DenseMatrix.diagonal(3, 0.0)
        ger(1.0, sparse(3, 0 to 0.0, 1 to 1.0), sparse(3, 2 to 5.0), sparse)
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == 1 && j == 2) 5.0 else 0.0, sparse[i, j], "sparse[$i,$j]")
            }
        }
    }

    @Test
    fun `the ops reject mismatched sizes and shapes`() {
        assertFailsWith<IllegalArgumentException> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<IllegalArgumentException> { axpy(dense(1.0), 1.0, dense(1.0, 2.0)) }
        assertFailsWith<IllegalArgumentException> {
            ger(1.0, dense(1.0, 2.0, 3.0), dense(1.0, 2.0), DenseMatrix(2, 2))
        }
        assertFailsWith<IllegalArgumentException> { gemv(DenseMatrix(2, 3), dense(1.0, 2.0)) }
    }

    @Test
    fun `transpose round-trips and maps entries`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
            ),
        )
        val t = a.transpose()
        assertEquals(3, t.rows)
        assertEquals(2, t.cols)
        for (i in 0 until a.rows) for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i])
        assertEquals(a, t.transpose())
        // Degenerate shapes.
        assertEquals(DenseMatrix(0, 0), DenseMatrix(0, 0).transpose())
        assertEquals(0, DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `transpose agrees with the gemm transpose flag`() {
        val rng = Random(20260804)
        val a = randomMatrix(4, 6, rng)
        val b = randomMatrix(4, 3, rng)
        // A-transpose times B, via a materialized transpose against the flag.
        val viaMaterialized = a.transpose().matMul(b)
        val viaFlag = DenseMatrix(6, 3)
        koblas.gemm(1.0, a, true, b, false, 0.0, viaFlag)
        assertClose(viaMaterialized, viaFlag, "transpose flag vs materialized")
    }
}
