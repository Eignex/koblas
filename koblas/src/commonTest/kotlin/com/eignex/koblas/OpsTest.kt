@file:Suppress("PropertyName") // math convention: single-letter matrices (A, M) in tests

package com.eignex.koblas

import com.eignex.koblas.core.*
import com.eignex.koblas.dense.Uplo
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@Suppress("VariableNaming") // single-letter matrix/vector names track math conventions
class OpsTest {

    private val sparse = F64SparseVector.of(6, intArrayOf(4, 1), doubleArrayOf(-3.0, 2.0))
    private val denseOfSparse = F64DenseVector.of(doubleArrayOf(0.0, 2.0, 0.0, 0.0, -3.0, 0.0))

    private fun dense(vararg v: Double) = F64DenseVector.of(v)
    private fun sparse(size: Int, vararg pairs: Pair<Int, Double>) =
        F64SparseVector.of(size, pairs.map { it.first }.toIntArray(), pairs.map { it.second }.toDoubleArray())

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
        val y = F64DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        y.axpy(2.0, sparse(3, 0 to 1.0, 2 to -1.0))
        assertEquals(dense(3.0, 2.0, 1.0), y)
    }

    @Test
    fun `scale mutates in place and respects identity`() {
        val v = F64DenseVector.of(doubleArrayOf(1.0, -2.0, 3.0))
        v.scale(0.5)
        assertEquals(dense(0.5, -1.0, 1.5), v)
        v.scale(1.0) // no-op
        assertEquals(dense(0.5, -1.0, 1.5), v)
    }

    @Test
    fun `norm2 matches the hand value on dense and sparse`() {
        assertEquals(5.0, F64DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0)).norm2())
        assertEquals(sqrt(13.0), sparse.norm2(), 1e-15)
        assertEquals(0.0, F64DenseVector.zero(4).norm2())
        assertEquals(0.0, F64DenseVector.zero(0).norm2())
    }

    @Test
    fun `norm2 survives overflow and underflow via the rescale fallback`() {
        assertEquals(5.0e200, F64DenseVector.of(doubleArrayOf(3.0e200, 0.0, -4.0e200)).norm2(), 1e186)
        assertEquals(5.0e-200, F64DenseVector.of(doubleArrayOf(3.0e-200, 4.0e-200)).norm2(), 1e-214)
        assertEquals(1.0e-300, F64DenseVector.of(doubleArrayOf(1.0e-300)).norm2(), 1e-314)
        val sparseHuge = F64SparseVector.of(5, intArrayOf(0, 3), doubleArrayOf(3.0e200, 4.0e200))
        assertEquals(5.0e200, sparseHuge.norm2(), 1e186)
        assertTrue(F64DenseVector.of(doubleArrayOf(1.0, Double.NaN)).norm2().isNaN())
        assertTrue(F64DenseVector.of(doubleArrayOf(1.0e200, Double.NaN)).norm2().isNaN())
        assertEquals(Double.POSITIVE_INFINITY, F64DenseVector.of(doubleArrayOf(1.0, Double.NEGATIVE_INFINITY)).norm2())
    }

    @Test
    fun `asum matches the hand value on dense and sparse`() {
        assertEquals(7.0, F64DenseVector.of(doubleArrayOf(3.0, 0.0, -4.0)).asum())
        assertEquals(5.0, sparse.asum())
        assertEquals(0.0, F64DenseVector.zero(0).asum())
    }

    @Test
    fun `norm1 is the maximum absolute column sum`() {
        val a = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, 5.0, -6.0),
            ),
        )
        assertEquals(9.0, a.norm1()) // columns sum to 5, 7, 9
        assertEquals(0.0, F64DenseMatrix(0, 3).norm1())
        assertEquals(0.0, F64DenseMatrix(3, 0).norm1())
    }

    @Test
    fun `iamax returns the first maximal index and handles edge cases`() {
        assertEquals(2, F64DenseVector.of(doubleArrayOf(1.0, -2.0, 5.0, -5.0)).iamax())
        assertEquals(1, F64DenseVector.of(doubleArrayOf(1.0, -5.0, 5.0)).iamax()) // tie: first wins
        assertEquals(4, sparse.iamax())
        assertEquals(0, F64DenseVector.zero(3).iamax()) // zero vector: first element
        assertEquals(0, F64SparseVector.of(3, IntArray(0), DoubleArray(0)).iamax()) // all-unstored: same
        assertEquals(-1, F64DenseVector.zero(0).iamax())
    }

    @Test
    fun `iamax reads a stored zero as the zero it is`() {
        assertEquals(0, F64SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(0, F64SparseVector.of(5, intArrayOf(1, 3), doubleArrayOf(0.0, -0.0)).iamax())
        assertEquals(F64DenseVector.zero(5).iamax(), F64SparseVector.of(5, intArrayOf(3), doubleArrayOf(0.0)).iamax())
        assertEquals(4, F64SparseVector.of(5, intArrayOf(1, 4), doubleArrayOf(0.0, -2.0)).iamax())
    }

    @Test
    fun `copy replicates dense and sparse sources and rejects size mismatch`() {
        val dst = F64DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0, 9.0, 9.0, 9.0))
        copy(sparse, dst) // sparse: must zero-fill the unstored slots
        assertContentEquals(denseOfSparse.data, dst.data)
        copy(F64DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)), dst)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dst.data)
        assertFailsWith<IllegalArgumentException> { copy(F64DenseVector.zero(2), F64DenseVector.zero(3)) }
    }

    @Test
    fun `swap exchanges contents and rejects size mismatch`() {
        val a = F64DenseVector.of(doubleArrayOf(1.0, 2.0))
        val b = F64DenseVector.of(doubleArrayOf(3.0, 4.0))
        swap(a, b)
        assertContentEquals(doubleArrayOf(3.0, 4.0), a.data)
        assertContentEquals(doubleArrayOf(1.0, 2.0), b.data)
        assertFailsWith<IllegalArgumentException> { swap(F64DenseVector.zero(2), F64DenseVector.zero(3)) }
    }

    @Test
    fun `level-1 ops agree with naive references on random vectors`() {
        val rng = Random(20260727)
        repeat(20) {
            val n = rng.nextInt(1, 200)
            val data = DoubleArray(n) { rng.nextDouble(-100.0, 100.0) }
            val v = F64DenseVector.of(data)
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

    @Test
    fun `the gemv overload computes A x for dense and sparse x`() {
        val A = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0),
                doubleArrayOf(5.0, 6.0),
            ),
        )
        val xDense = dense(1.0, -1.0)
        val xSparse = sparse(2, 0 to 1.0, 1 to -1.0)
        val expected = dense(1.0 * 1 + 2 * -1, 3.0 * 1 + 4 * -1, 5.0 * 1 + 6 * -1)
        assertEquals(expected, A.matVec(xDense))
        assertEquals(expected, A.matVec(xSparse))
    }

    @Test
    fun `sparse and dense gemv agree on a random example`() {
        val rng = Random(7)
        val n = 8
        val A = randomMatrix(n, n, rng)
        val nz = (0 until n).filter { rng.nextBoolean() }
        val xv = DoubleArray(n)
        for (i in nz) xv[i] = rng.nextDouble(-1.0, 1.0)
        val xSparse = F64SparseVector.of(n, nz.toIntArray(), nz.map { xv[it] }.toDoubleArray())
        assertClose(A.matVec(F64DenseVector.of(xv)).data, A.matVec(xSparse).data, "matVec dense vs sparse")
    }

    @Test
    fun `ger updates a matrix with alpha x y_transpose`() {
        val M = F64DenseMatrix.diagonal(2, 1.0)
        M.ger(0.5, dense(1.0, 2.0), dense(3.0, 4.0))
        assertEquals(1.0 + 0.5 * 3, M[0, 0], 1e-12)
        assertEquals(0.5 * 4, M[0, 1], 1e-12)
        assertEquals(0.5 * 6, M[1, 0], 1e-12)
        assertEquals(1.0 + 0.5 * 8, M[1, 1], 1e-12)
    }

    @Test
    fun `ger with sparse operands only touches nonzero rows and cols`() {
        val M = F64DenseMatrix.diagonal(3, 0.0)
        M.ger(1.0, sparse(3, 1 to 2.0), sparse(3, 0 to 3.0, 2 to 4.0))
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
        val y = F64DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0))
        y.axpy(0.0, F64DenseVector.of(doubleArrayOf(9.0, 9.0, 9.0)))
        assertTrue(y.toDoubleArray().contentEquals(doubleArrayOf(1.0, 2.0, 3.0)))
        val M = F64DenseMatrix.diagonal(2, 1.0)
        M.ger(0.0, F64DenseVector.of(doubleArrayOf(1.0, 1.0)), F64DenseVector.of(doubleArrayOf(1.0, 1.0)))
        for (i in 0 until 2) {
            for (j in 0 until 2) assertEquals(if (i == j) 1.0 else 0.0, M[i, j])
        }
    }

    @Test
    fun `ger skips zero entries on both carriers`() {
        val dense = F64DenseMatrix.diagonal(3, 0.0)
        dense.ger(1.0, dense(0.0, 2.0, 0.0), dense(1.0, 1.0, 1.0))
        for (i in 0 until 3) {
            for (j in 0 until 3) assertEquals(if (i == 1) 2.0 else 0.0, dense[i, j], 1e-12, "dense[$i,$j]")
        }
        val sparse = F64DenseMatrix.diagonal(3, 0.0)
        sparse.ger(1.0, sparse(3, 0 to 0.0, 1 to 1.0), sparse(3, 2 to 5.0))
        for (i in 0 until 3) {
            for (j in 0 until 3) {
                assertEquals(if (i == 1 && j == 2) 5.0 else 0.0, sparse[i, j], "sparse[$i,$j]")
            }
        }
    }

    @Test
    fun `syr and syr2 match the equivalent ger sweeps`() {
        val rng = Random(20260807)
        val n = 6
        val x = F64DenseVector.of(randomVector(n, rng))
        val y = F64DenseVector.of(randomVector(n, rng))

        val viaSyr = F64DenseMatrix(n, n)
        viaSyr.syr(1.5, x)
        val viaGer = F64DenseMatrix(n, n)
        viaGer.ger(1.5, x, x)
        assertClose(viaGer, viaSyr, "syr against ger")

        val viaSyr2 = F64DenseMatrix(n, n)
        viaSyr2.syr2(-0.75, x, y)
        val viaGer2 = F64DenseMatrix(n, n)
        viaGer2.ger(-0.75, x, y)
        viaGer2.ger(-0.75, y, x)
        assertClose(viaGer2, viaSyr2, "syr2 against two gers")
    }

    @Test
    fun `the symmetric updates are exactly symmetric and honour uplo`() {
        val rng = Random(20260808)
        val n = 5
        val x = F64DenseVector.of(randomVector(n, rng))
        val y = F64DenseVector.of(randomVector(n, rng))

        val full = F64DenseMatrix(n, n)
        full.syr2(0.3, x, y)
        for (i in 0 until n) {
            for (j in 0 until n) {
                assertTrue(full[i, j] == full[j, i], "syr2 FULL is not exactly symmetric at [$i,$j]")
            }
        }

        val lower = F64DenseMatrix(n, n)
        lower.syr(1.0, x, Uplo.LOWER)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i < j) assertEquals(0.0, lower[i, j], "syr LOWER wrote the upper triangle at [$i,$j]")
                if (i >= j) assertEquals(x[i] * x[j], lower[i, j], 1e-12, "[$i,$j]")
            }
        }
    }

    @Test
    fun `the ops reject mismatched sizes and shapes`() {
        assertFailsWith<IllegalArgumentException> { dense(1.0) dot dense(1.0, 2.0) }
        assertFailsWith<IllegalArgumentException> { dense(1.0).axpy(1.0, dense(1.0, 2.0)) }
        assertFailsWith<IllegalArgumentException> {
            F64DenseMatrix(2, 2).ger(1.0, dense(1.0, 2.0, 3.0), dense(1.0, 2.0))
        }
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(2, 3).matVec(dense(1.0, 2.0)) }
    }

    @Test
    fun `transpose round-trips and maps entries`() {
        val a = F64DenseMatrix.of(
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
        assertEquals(F64DenseMatrix(0, 0), F64DenseMatrix(0, 0).transpose())
        assertEquals(0, F64DenseMatrix(0, 5).transpose().cols)
        assertEquals(5, F64DenseMatrix(0, 5).transpose().rows)
    }

    @Test
    fun `transpose agrees with the gemm transpose flag`() {
        val rng = Random(20260804)
        val a = randomMatrix(4, 6, rng)
        val b = randomMatrix(4, 3, rng)
        val viaMaterialized = a.transpose() * b
        val viaFlag = F64DenseMatrix(6, 3)
        koblas.gemm(1.0, a, true, b, false, 0.0, viaFlag)
        assertClose(viaMaterialized, viaFlag, "transpose flag vs materialized")
    }

    @Test
    fun `sparse against sparse dot matches the dense answer over merge shapes`() {
        val n = 8
        val patterns = listOf(
            intArrayOf(0, 2, 4, 6) to intArrayOf(1, 3, 5, 7), // disjoint, fully interleaved
            intArrayOf(0, 1, 2) to intArrayOf(0, 1, 2), // identical
            intArrayOf(0, 7) to intArrayOf(3, 4), // disjoint, nested inside the first span
            intArrayOf(0, 1, 2, 3) to intArrayOf(3), // one side exhausts early
            intArrayOf(5) to intArrayOf(0, 1, 2, 3, 4), // the other side does
            IntArray(0) to intArrayOf(0, 4), // empty against non-empty
            IntArray(0) to IntArray(0),
        )
        for ((ia, ib) in patterns) {
            val a = F64SparseVector.of(n, ia, DoubleArray(ia.size) { it + 1.5 })
            val b = F64SparseVector.of(n, ib, DoubleArray(ib.size) { it + 2.5 })
            val expected = F64DenseVector.of(a.toDoubleArray()) dot F64DenseVector.of(b.toDoubleArray())
            assertEquals(expected, a dot b, "pattern ${ia.toList()} vs ${ib.toList()}")
            assertEquals(expected, b dot a, "dot should be symmetric for ${ia.toList()} vs ${ib.toList()}")
        }
    }

    @Test
    fun `mixed sparse and dense dot agrees in both operand orders`() {
        val sparse = F64SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val dense = F64DenseVector.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        val expected = 2.0 * 2.0 + -3.0 * 5.0
        assertEquals(expected, sparse dot dense)
        assertEquals(expected, dense dot sparse)
    }
}
