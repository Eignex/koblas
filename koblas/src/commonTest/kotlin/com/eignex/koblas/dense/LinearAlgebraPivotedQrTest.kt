package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * QR with column pivoting: that it factorizes the *permuted* matrix, that the rank it reports is the rank,
 * and that the solve undoes the permutation.
 *
 * Rank is checked on matrices whose rank is known by construction, built as products of full-rank factors
 * with a chosen inner dimension, rather than on matrices that merely look dependent. Both the exactly
 * dependent case and the nearly dependent one are covered, because they fail differently: exact dependence
 * leaves a zero on the diagonal, while near dependence leaves something small that only a tolerance can
 * judge.
 */
class LinearAlgebraPivotedQrTest {

    /** An `m×n` matrix of rank exactly [rank], as a product of two random full-rank factors. */
    private fun ofRank(m: Int, n: Int, rank: Int, rng: Random): DenseMatrix {
        val left = randomMatrix(m, rank, rng)
        val right = randomMatrix(rank, n, rng)
        val a = DenseMatrix(m, n)
        for (i in 0 until m) {
            for (j in 0 until n) {
                var s = 0.0
                for (p in 0 until rank) s += left[i, p] * right[p, j]
                a[i, j] = s
            }
        }
        return a
    }

    /** `A·P` — the matrix the factorization actually decomposes. */
    private fun permuted(a: DenseMatrix, pivots: IntArray): DenseMatrix {
        val out = DenseMatrix(a.rows, a.cols)
        for (j in pivots.indices) for (i in 0 until a.rows) out[i, j] = a[i, pivots[j]]
        return out
    }

    /** Column [j] of `R`, padded to length `m`. */
    private fun rColumn(qr: QrDecomposition, j: Int): DoubleArray {
        val col = DoubleArray(qr.m)
        for (i in 0 until minOf(j + 1, qr.tau.size)) col[i] = qr.qr[i + j * qr.m]
        return col
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
        // Pivoting always takes the largest remaining column, which is exactly what makes the rank count a
        // leading run rather than a scan.
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

    /**
     * Dependence in the *leading* columns, which is the case that separates pivoted QR from plain QR.
     *
     * The rank is the leading run of diagonal entries above the tolerance, so a factorization that took the
     * columns in their given order would stop at position 1 and report rank 1 for a matrix of rank 3. Only
     * moving the dependent column to the end recovers the real answer, so this is the test that fails if the
     * pivot search is removed — the rank tests above pass either way, because dependence built from random
     * factors lands at the end on its own.
     */
    @Test
    fun `dependence among the leading columns is pivoted to the end`() {
        val rng = Random(31337)
        val a = randomMatrix(6, 4, rng)
        for (i in 0 until 6) a[i, 1] = a[i, 0]
        val f = koblas.qrPivoted(a)
        assertEquals(3, f.rank, "a duplicate in column 1 must not truncate the rank to 1")
        assertTrue(f.rankDeficient)
        // One of the two identical columns must end up in the trailing position the rank excludes.
        assertTrue(f.pivots.last() == 0 || f.pivots.last() == 1, "the dependent column belongs last")
    }

    /**
     * A column that loses nearly all of its norm in one step, which is what the norm downdating has to
     * survive.
     *
     * Pivoted QR keeps a running norm per column and shrinks it by the component each reflector removes,
     * because recomputing every trailing norm at every step would cost an extra `O(m·n²)`. When a column is
     * almost entirely in the direction just eliminated, that subtraction is between two numbers equal to
     * within rounding and the result collapses to zero — so both columns here would be recorded as exhausted
     * and the pivot search would take them in the order they arrived, which is the wrong one. LAPACK's guard,
     * which this reproduces, notices the collapse and recomputes the norm from the column itself.
     *
     * The observable consequence is the rank. Taken in the right order the tiny column lands last, below the
     * tolerance, and the rank is 2; taken in arrival order the `1e-17` column lands second and truncates the
     * leading run to 1.
     */
    @Test
    fun `a column that collapses in one step keeps its true norm`() {
        val a = DenseMatrix(3, 3)
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
        // Column 2 differs from column 0 by 1e-10, which is far above the automatic tolerance and far below
        // a caller who considers anything under 1e-6 to be the same column.
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
        // Same problem, same minimiser: if the permutation were not undone these would be a shuffle apart.
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
        // The residual must be orthogonal to every column of A, which is what makes it a minimiser — and is
        // the property a plausible-looking wrong answer fails.
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
        val empty = koblas.qrPivoted(DenseMatrix(0, 0))
        assertEquals(0, empty.rank)
        assertEquals(0, empty.pivots.size)
        val zeros = koblas.qrPivoted(DenseMatrix(4, 3))
        assertEquals(0, zeros.rank, "an all-zero matrix has rank zero")
        assertEquals(listOf(0, 1, 2), zeros.pivots.sorted())
    }

    @Test
    fun `the solve checks its shapes`() {
        val f = koblas.qrPivoted(randomMatrix(6, 3, Random(8)))
        assertFailsWith<IllegalArgumentException> { koblas.solveLeastSquares(f, DoubleArray(5)) }
        assertFailsWith<IllegalArgumentException> {
            koblas.solveLeastSquaresInto(f, DoubleArray(6), DoubleArray(2))
        }
    }
}
