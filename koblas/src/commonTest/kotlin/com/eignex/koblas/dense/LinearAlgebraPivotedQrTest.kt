package com.eignex.koblas.dense

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.host.rColumn
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LinearAlgebraPivotedQrTest {

    private fun ofRank(m: Int, n: Int, rank: Int, rng: Random): F64DenseMatrix {
        val left = randomMatrix(m, rank, rng)
        val right = randomMatrix(rank, n, rng)
        val a = F64DenseMatrix(m, n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var s = 0.0
                for (p in 0 until rank) s += left[i, p] * right[p, j]
                a[i, j] = s
            }
        }
        return a
    }

    private fun permuted(a: F64DenseMatrix, pivots: IntArray): F64DenseMatrix {
        val out = F64DenseMatrix(a.rows, a.cols)
        for (j in pivots.indices) for (i in 0 until a.rows) out[i, j] = a[i, pivots[j]]
        return out
    }

    @Test
    fun `Q times R reconstructs the permuted matrix`() {
        val rng = Random(20260807)
        for ((m, n) in listOf(1 to 1, 5 to 5, 9 to 4, 4 to 9, 12 to 7)) {
            val a = randomMatrix(m, n, rng)
            val f = koblas.qrPivoted(a)
            val expected = permuted(a, f.pivots)
            for (j in 0 until n) {
                val rebuilt = koblas.applyQ(f.factorization, rColumn(f.factorization, j))
                for (i in 0 until m) {
                    assertClose(expected[i, j], rebuilt[i], "Q·R at ($i,$j) for ${m}x$n", tolerance = 1e-9)
                }
            }
        }
    }

    @Test
    fun `the pivots are a permutation and the R diagonal is non-increasing`() {
        val rng = Random(4242)
        val a = randomMatrix(10, 6, rng)
        val f = koblas.qrPivoted(a)
        assertEquals((0 until 6).toList(), f.pivots.sorted(), "pivots must be a permutation of the columns")
        var previous = Double.MAX_VALUE
        for (d in 0 until 6) {
            val entry = abs(f.factorization.qr[d + d * 10])
            assertTrue(entry <= previous * (1.0 + 1e-12), "R diagonal rose at $d: $entry after $previous")
            previous = entry
        }
    }

    @Test
    fun `rank matches the construction for exactly dependent columns`() {
        val rng = Random(77)
        for (rank in 1..4) {
            val a = ofRank(8, 6, rank, rng)
            assertEquals(rank, koblas.qrPivoted(a).rank, "rank of an 8x6 built at rank $rank")
        }
        assertEquals(6, koblas.qrPivoted(randomMatrix(8, 6, rng)).rank, "a random tall matrix has full rank")
    }

    @Test
    fun `dependence among the leading columns is pivoted to the end`() {
        val rng = Random(31337)
        val a = randomMatrix(6, 4, rng)
        for (i in 0 until 6) a[i, 1] = a[i, 0]
        val f = koblas.qrPivoted(a)
        assertEquals(3, f.rank, "a duplicate in column 1 must not truncate the rank to 1")
        assertTrue(f.rankDeficient)
        assertTrue(f.pivots.last() == 0 || f.pivots.last() == 1, "the dependent column belongs last")
    }

    @Test
    fun `a column that collapses in one step keeps its true norm`() {
        val a = F64DenseMatrix(3, 3)
        a[0, 0] = 1.0
        a[0, 1] = 1.0
        a[2, 1] = 1e-17 // below the tolerance: a direction that is not really there
        a[0, 2] = 1.0
        a[1, 2] = 1e-9 // above it: a real, very small direction
        val f = koblas.qrPivoted(a)
        assertEquals(2, f.rank, "the 1e-9 direction is real and must be found before the 1e-17 one")
        assertEquals(1, f.pivots.last(), "the column that is not really a direction belongs last")
    }

    @Test
    fun `near dependence is the tolerance's decision`() {
        val rng = Random(555)
        val a = randomMatrix(7, 3, rng)
        // Column 2 differs from column 0 by 1e-10, above the automatic tolerance and below a tolerance of 1e-6.
        for (i in 0 until 7) a[i, 2] = a[i, 0] + 1e-10 * rng.nextDouble()
        assertEquals(3, koblas.qrPivoted(a).rank, "the default tolerance sees a real third direction")
        assertEquals(2, koblas.qrPivoted(a, tolerance = 1e-6).rank, "a looser tolerance calls it dependent")
    }

    @Test
    fun `the least-squares solve undoes the permutation`() {
        val rng = Random(90210)
        val a = randomMatrix(9, 4, rng)
        val b = randomVector(9, rng)
        val pivoted = koblas.solveLeastSquares(koblas.qrPivoted(a), b)
        val plain = koblas.solveLeastSquares(koblas.qr(a), b)
        assertClose(plain, pivoted, "pivoted against unpivoted least squares", tolerance = 1e-8)
    }

    @Test
    fun `the rank-deficient solve is a basic solution that still minimises the residual`() {
        val rng = Random(1234)
        val a = ofRank(10, 5, 3, rng)
        val b = randomVector(10, rng)
        val f = koblas.qrPivoted(a)
        assertEquals(3, f.rank)
        val x = koblas.solveLeastSquares(f, b)
        assertEquals(2, x.count { it == 0.0 }, "a basic solution leaves n − rank entries at zero")
        val residual = DoubleArray(10)
        for (i in 0 until 10) {
            var s = 0.0
            for (j in 0 until 5) s += a[i, j] * x[j]
            residual[i] = s - b[i]
        }
        for (j in 0 until 5) {
            var s = 0.0
            for (i in 0 until 10) s += a[i, j] * residual[i]
            assertTrue(abs(s) < 1e-8, "residual not orthogonal to column $j: $s")
        }
    }

    @Test
    fun `degenerate shapes factor without special-casing`() {
        val empty = koblas.qrPivoted(F64DenseMatrix(0, 0))
        assertEquals(0, empty.rank)
        assertEquals(0, empty.pivots.size)
        val zeros = koblas.qrPivoted(F64DenseMatrix(4, 3))
        assertEquals(0, zeros.rank, "an all-zero matrix has rank zero")
        assertEquals(listOf(0, 1, 2), zeros.pivots.sorted())
        // A matrix with no rows still has columns, and they still have to come back as a permutation.
        val noRows = koblas.qrPivoted(F64DenseMatrix(0, 3))
        assertEquals(0, noRows.rank, "a matrix with no rows has rank zero")
        assertEquals(listOf(0, 1, 2), noRows.pivots.sorted(), "a zero-row matrix must still permute its columns")
        val noColumns = koblas.qrPivoted(F64DenseMatrix(4, 0))
        assertEquals(0, noColumns.rank, "a matrix with no columns has rank zero")
        assertEquals(0, noColumns.pivots.size)
    }

    @Test
    fun `the solve checks its shapes`() {
        val f = koblas.qrPivoted(randomMatrix(6, 3, Random(8)))
        assertFailsWith<DimensionMismatch> { koblas.solveLeastSquares(f, DoubleArray(5)) }
        assertFailsWith<IllegalArgumentException> {
            koblas.solveLeastSquaresInto(f, DoubleArray(6), DoubleArray(2))
        }
    }

    @Test
    fun `a negative tolerance is rejected rather than read as automatic`() {
        val a = randomMatrix(6, 3, Random(20260919))
        for (tolerance in doubleArrayOf(-1.0, -1e-300, Double.NEGATIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException>("tolerance $tolerance should be rejected") {
                koblas.qrPivoted(a, tolerance)
            }
        }
        // Zero still means the shape-derived default, and a positive value is still honoured.
        assertEquals(3, koblas.qrPivoted(a, AUTOMATIC_RANK_TOLERANCE).rank)
        assertEquals(3, koblas.qrPivoted(a, 1e-12).rank)
    }
}
