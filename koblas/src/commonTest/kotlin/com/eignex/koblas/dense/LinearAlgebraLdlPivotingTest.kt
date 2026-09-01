package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The Bunch-Kaufman paths a dominant diagonal never reaches: 2x2 blocks with a trailing update behind them,
 * and the row and column interchanges that carry a pivot in from further down. A matrix with no diagonal to
 * speak of is what forces them, so every matrix here has a zero or tiny diagonal.
 */
class LinearAlgebraLdlPivotingTest {

    /** Symmetric with a [diagonalScale] diagonal against unit off-diagonals, poisoned above the diagonal. */
    private fun poisoned(rng: Random, n: Int, diagonalScale: Double): Pair<F64DenseMatrix, F64DenseMatrix> {
        val full = F64DenseMatrix(n)
        val lowerOnly = F64DenseMatrix(n)
        for (i in 0 until n) {
            for (j in 0..i) {
                val v = if (i == j) diagonalScale * rng.nextDouble(-1.0, 1.0) else rng.nextDouble(-1.0, 1.0)
                full[i, j] = v
                full[j, i] = v
                lowerOnly[i, j] = v
                if (i != j) lowerOnly[j, i] = Double.NaN
            }
        }
        return full to lowerOnly
    }

    /** How many 2x2 blocks the factorization chose, so a test can assert it exercised that path. */
    private fun twoByTwoBlocks(blocks: List<F64PivotedSymmetricIndefinitePivotBlock>): Int =
        blocks.count { it is F64PivotedSymmetricIndefinitePivotBlock.TwoByTwo }

    private fun residual(a: F64DenseMatrix, x: DoubleArray, b: DoubleArray): Double {
        val ax = koblas.gemv(a, x)
        var worst = 0.0
        for (i in b.indices) worst = maxOf(worst, abs(ax[i] - b[i]))
        return worst
    }

    /**
     * A vanishing diagonal leaves no acceptable 1x1 pivot, so the factorization has to take 2x2 blocks and,
     * from n = 3 up, run the trailing update behind them.
     */
    @Test
    fun `a vanishing diagonal factors through two-by-two blocks and still solves`() {
        val rng = Random(20260817)
        var blocksSeen = 0
        for (n in intArrayOf(2, 3, 4, 5, 8, 13, 20)) {
            val (full, lowerOnly) = poisoned(rng, n, diagonalScale = 0.0)
            val f = koblas.pivotedSymmetricIndefinite(lowerOnly)
            assertTrue(!f.singular, "n=$n flagged singular")
            blocksSeen += twoByTwoBlocks(f.pivotBlocks)
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val x = koblas.solve(f, b)
            assertTrue(
                residual(full, x, b) <= 1e-8,
                "n=$n residual ${residual(full, x, b)} for a zero-diagonal matrix",
            )
        }
        assertTrue(blocksSeen > 0, "no 2x2 block was taken, so the pivoting path under test never ran")
    }

    /**
     * A diagonal small but not zero is where the three-way Bunch-Kaufman test actually branches, taking a
     * 1x1 pivot from further down the column as often as a 2x2 block.
     */
    @Test
    fun `a small diagonal exercises the interchange and both pivot sizes`() {
        var blocksSeen = 0
        var interchanges = 0
        for (seed in 0 until 40) {
            val rng = Random(20260818 + seed)
            val n = 4 + seed % 9
            val (full, lowerOnly) = poisoned(rng, n, diagonalScale = 0.05)
            val f = koblas.pivotedSymmetricIndefinite(lowerOnly)
            if (f.singular) continue
            val blocks = f.pivotBlocks
            blocksSeen += twoByTwoBlocks(blocks)
            interchanges += blocks.count { it.interchangePosition != it.interchangedWith }
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val x = koblas.solve(f, b)
            assertTrue(residual(full, x, b) <= 1e-7, "seed=$seed n=$n residual ${residual(full, x, b)}")
        }
        assertTrue(blocksSeen > 0, "no 2x2 block was taken across 40 matrices")
        assertTrue(interchanges > 0, "no pivot was ever brought in from another row")
    }

    /**
     * The pivot two rows or more below the diagonal, which is the only way to reach the middle interchange
     * loop that swaps a row segment against a column segment.
     */
    @Test
    fun `a pivot far below the diagonal interchanges correctly`() {
        val n = 5
        val full = F64DenseMatrix(n)
        val lowerOnly = F64DenseMatrix(n)
        // Zero diagonal, and column 0's only large entry is in the last row, so the pivot comes from there.
        for (i in 0 until n) {
            for (j in 0..i) {
                val v = when {
                    i == j -> 0.0
                    j == 0 && i == n - 1 -> 4.0
                    else -> 0.25
                }
                full[i, j] = v
                full[j, i] = v
                lowerOnly[i, j] = v
                if (i != j) lowerOnly[j, i] = Double.NaN
            }
        }
        val f = koblas.pivotedSymmetricIndefinite(lowerOnly)
        assertTrue(!f.singular, "the interchange case flagged singular")
        val b = DoubleArray(n) { 1.0 + it }
        val x = koblas.solve(f, b)
        assertTrue(residual(full, x, b) <= 1e-8, "residual ${residual(full, x, b)}")
    }

    /**
     * A subnormal pivot is a valid 1x1 pivot, and its reciprocal overflows where dividing by it does not.
     * The multiplier is what discriminates: `a(1,0) / a(0,0)` is exactly 1 for this matrix, where scaling by
     * the reciprocal gives an infinity and skipping the scaling altogether leaves the raw entry behind.
     *
     * Against the reference by name, not the installed backend. This is the one place the reference departs
     * from `dsytf2`, so a host LAPACKE answers it with an infinity, and only the JVM pins the backend.
     */
    @Test
    fun `a subnormal pivot divides the column instead of scaling it`() {
        val tiny = 1e-320
        val n = 2
        val lowerOnly = F64DenseMatrix(n)
        lowerOnly[0, 0] = tiny
        lowerOnly[1, 0] = tiny
        lowerOnly[1, 1] = 1.0
        lowerOnly[0, 1] = Double.NaN
        val f = F64ReferenceLinearAlgebra.pivotedSymmetricIndefinite(lowerOnly)
        assertTrue(!f.singular, "a subnormal pivot is not a zero pivot")
        assertEquals(
            1.0,
            f.ldl[1],
            "multiplier is ${f.ldl[1]}, expected exactly 1.0",
        )
        for (i in 0 until n) {
            for (j in 0..i) {
                assertTrue(
                    f.ldl[i + j * n].isFinite(),
                    "factor entry ($i;$j) is ${f.ldl[i + j * n]}, so the pivot reciprocal overflowed",
                )
            }
        }
    }

    /** Multi-RHS goes column by column, so it has to agree with the vector solve on the same matrix. */
    @Test
    fun `the multi-RHS solve agrees with the vector solve through a two-by-two block`() {
        val rng = Random(20260819)
        val n = 7
        val nrhs = 3
        val (_, lowerOnly) = poisoned(rng, n, diagonalScale = 0.0)
        val f = koblas.pivotedSymmetricIndefinite(lowerOnly)
        assertTrue(twoByTwoBlocks(f.pivotBlocks) > 0, "expected a 2x2 block")
        val b = F64DenseMatrix(n, nrhs)
        for (idx in b.data.indices) b.data[idx] = rng.nextDouble(-1.0, 1.0)
        val block = koblas.solve(f, b)
        for (c in 0 until nrhs) {
            val column = DoubleArray(n) { i -> b[i, c] }
            val expected = koblas.solve(f, column)
            val actual = DoubleArray(n) { i -> block[i, c] }
            assertClose(expected, actual, "multi-RHS column $c", tolerance = 1e-12)
        }
    }
}
