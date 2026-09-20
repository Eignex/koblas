@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.StridedVector
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A backend that recommends a grouping of its own, so the shared traversal can be run at widths no production
 * backend chooses.
 *
 * The arithmetic is the portable backend's, because what is under test here is the scheduling: the windows a
 * triangular or symmetric traversal cuts, the corner it keeps for itself, and the tail at the end of a panel
 * all move with the grouping, and every one of them has to come out the same.
 */
private class RegroupedPanels(private val group: Int) : DensePanelKernels by PortablePanelKernels {
    override val name: String get() = "regrouped($group)"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = group
}

/**
 * The shared dense scheduling, at every grouping a backend might recommend.
 *
 * One backend's recommendation is one point of this sweep, so a traversal that happened to be right at four
 * columns and wrong at three would pass on this machine and fail on the next one. The oracle is
 * [ReferenceBlas], which is the textbook definition and shares no code with the scheduling.
 */
class PortableDenseBlasTest {
    private val groups = intArrayOf(1, 2, 3, 5, 7, 64)

    private fun blasFor(group: Int): DenseBlas = PortableDenseBlas(ScalarVectorKernels, RegroupedPanels(group))

    private fun orders() = intArrayOf(0, 1, 2, 3, 5, 8, 9, 17)

    @Test
    fun `gemv agrees with the reference at every grouping and orientation`() {
        val rng = Random(20260921)
        for (group in groups) {
            val blas = blasFor(group)
            for (rows in orders()) {
                val cols = if (rows == 0) 0 else rows + 2
                val a = randomMatrix(rows, cols, rng)
                for (transpose in booleanArrayOf(false, true)) {
                    val x = randomVector(if (transpose) rows else cols, rng)
                    val y0 = randomVector(if (transpose) cols else rows, rng)
                    for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                        val expected = y0.copyOf()
                        ReferenceBlas.gemv(0.875, a, x, beta, expected, transpose)
                        val actual = y0.copyOf()
                        blas.gemv(0.875, a, x, beta, actual, transpose)
                        assertClose(expected, actual, "gemv group=$group rows=$rows t=$transpose beta=$beta")
                    }
                }
            }
        }
    }

    @Test
    fun `symv agrees with the reference at every grouping and triangle`() {
        val rng = Random(20260922)
        for (group in groups) {
            val blas = blasFor(group)
            for (n in orders()) {
                val a = randomMatrix(n, n, rng)
                val x = randomVector(n, rng)
                val y0 = randomVector(n, rng)
                for (lower in booleanArrayOf(true, false)) {
                    val expected = y0.copyOf()
                    ReferenceBlas.symv(0.875, a, x, -0.25, expected, lower)
                    val actual = y0.copyOf()
                    blas.symv(0.875, a, x, -0.25, actual, lower)
                    assertClose(expected, actual, "symv group=$group n=$n lower=$lower")
                }
            }
        }
    }

    @Test
    fun `rank updates agree with the reference at every grouping and triangle`() {
        val rng = Random(20260923)
        for (group in groups) {
            val blas = blasFor(group)
            for (n in orders()) {
                val x = randomVector(n, rng)
                val y = randomVector(n, rng)
                val start = randomMatrix(n, n, rng)
                for (lower in booleanArrayOf(true, false)) {
                    val expected = copyOf(start)
                    ReferenceBlas.syr(0.875, DenseVector.wrap(x), expected, lower)
                    val actual = copyOf(start)
                    blas.syr(0.875, DenseVector.wrap(x), actual, lower)
                    assertClose(expected, actual, "syr group=$group n=$n lower=$lower")

                    val expectedTwo = copyOf(start)
                    ReferenceBlas.syr2(0.875, DenseVector.wrap(x), DenseVector.wrap(y), expectedTwo, lower)
                    val actualTwo = copyOf(start)
                    blas.syr2(0.875, DenseVector.wrap(x), DenseVector.wrap(y), actualTwo, lower)
                    assertClose(expectedTwo, actualTwo, "syr2 group=$group n=$n lower=$lower")
                }
                val expectedGer = copyOf(start)
                ReferenceBlas.ger(0.875, x, y, expectedGer)
                val actualGer = copyOf(start)
                blas.ger(0.875, x, y, actualGer)
                assertClose(expectedGer, actualGer, "ger group=$group n=$n")
            }
        }
    }

    /**
     * A rank update over a vector with a step, which reaches the panel without a copy and so has to address
     * its window rather than assume adjacent entries.
     */
    @Test
    fun `a strided rank update agrees with the reference`() {
        val rng = Random(20260924)
        val n = 9
        val blas = blasFor(3)
        val buffer = randomVector(2 * n + 1, rng)
        for (stride in intArrayOf(2, -2)) {
            val origin = if (stride > 0) 1 else 2 * n - 1
            val view = StridedVector(buffer, origin, n, stride)
            val dense = DenseVector.of(DoubleArray(n) { view[it] })
            for (lower in booleanArrayOf(true, false)) {
                val start = randomMatrix(n, n, rng)
                val expected = copyOf(start)
                ReferenceBlas.syr(0.875, dense, expected, lower)
                val actual = copyOf(start)
                blas.syr(0.875, view, actual, lower)
                assertClose(expected, actual, "strided syr stride=$stride lower=$lower")
            }
        }
    }

    @Test
    fun `triangular vector operations agree with the reference at every grouping`() {
        val rng = Random(20260925)
        for (group in groups) {
            val blas = blasFor(group)
            for (n in orders()) {
                val a = triangleFor(n, rng)
                val x0 = randomVector(n, rng)
                for (lower in booleanArrayOf(true, false)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unit in booleanArrayOf(false, true)) {
                            val context = "group=$group n=$n lower=$lower t=$transpose unit=$unit"
                            val expectedMultiply = x0.copyOf()
                            ReferenceBlas.trmv(a, expectedMultiply, lower, transpose, unit)
                            val actualMultiply = x0.copyOf()
                            blas.trmv(a, actualMultiply, lower, transpose, unit)
                            assertClose(expectedMultiply, actualMultiply, "trmv $context")

                            val expectedSolve = x0.copyOf()
                            ReferenceBlas.trsv(a, expectedSolve, lower, transpose, unit)
                            val actualSolve = x0.copyOf()
                            blas.trsv(a, actualSolve, lower, transpose, unit)
                            assertClose(expectedSolve, actualSolve, "trsv $context", tolerance = 1e-9)
                        }
                    }
                }
            }
        }
    }

    /**
     * Where alpha sits in a product, which is what an exceptional operand makes visible.
     *
     * The untransposed traversal scales each column's coefficient and then updates the destination, and the
     * transposed one scales the finished reduction; both are where the reference BLAS puts it, and the two
     * are not the same arithmetic. An infinite multiplier against a zero entry is NaN one way and an infinity
     * the other, so the choice is pinned here rather than left to whichever grouping came last.
     */
    @Test
    fun `alpha is scaled into the coefficient going down a column and into the sum going across one`() {
        val blas = blasFor(4)
        val a = DenseMatrix.wrap(2, 2, doubleArrayOf(0.0, 1.0, 1.0, 1.0))
        val x = doubleArrayOf(1.0, 1.0)

        val down = DoubleArray(2)
        blas.gemv(Double.POSITIVE_INFINITY, a, x, 0.0, down)
        val across = DoubleArray(2)
        blas.gemv(Double.POSITIVE_INFINITY, a, x, 0.0, across, transpose = true)

        // Down a column the coefficient is infinite before it meets the zero at (0, 0), which is a NaN.
        assertTrue(down[0].isNaN(), "an infinite coefficient against a zero entry was not NaN, got ${down[0]}")
        assertEquals(Double.POSITIVE_INFINITY, down[1])
        // Across a row the sum is finite and the multiplier reaches it afterwards, so it stays an infinity.
        assertEquals(Double.POSITIVE_INFINITY, across[0])
        assertEquals(Double.POSITIVE_INFINITY, across[1])
    }

    /**
     * A zero coefficient is still multiplied, because this is a matrix product and not an `axpy`.
     *
     * Both orientations evaluate every position the traversal reaches, so a zero against an infinity is the
     * NaN it evaluates to rather than a column skipped for being cheap.
     */
    @Test
    fun `a zero entry of x still multiplies an infinity in the matrix`() {
        // The infinity sits where the zero coefficient meets it in both directions: off the diagonal one way
        // and down the same column the other.
        val a = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1.0))
        val x = doubleArrayOf(1.0, 0.0)

        for (group in groups) {
            val blas = blasFor(group)
            val down = DoubleArray(2)
            blas.gemv(1.0, a, x, 0.0, down)
            val across = DoubleArray(2)
            blas.gemv(1.0, a, x, 0.0, across, transpose = true)

            assertTrue(down[0].isNaN(), "group $group skipped a zero coefficient going down a column")
            assertTrue(across[0].isNaN(), "group $group skipped a zero coefficient going across a row")
        }
    }

    /** A triangle with a dominant diagonal, so a solve over it is well conditioned at every order. */
    private fun triangleFor(n: Int, rng: Random): DenseMatrix {
        val a = randomMatrix(n, n, rng)
        for (i in 0 until n) a.values[i + i * n] = 2.0 + (i % 3)
        return a
    }

    private fun copyOf(a: DenseMatrix) = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())
}
