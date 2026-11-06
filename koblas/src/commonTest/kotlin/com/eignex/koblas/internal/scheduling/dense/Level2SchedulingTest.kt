package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.EngineImplementation
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.random.Random
import kotlin.test.*

/**
 * A backend that recommends a grouping of its own, so the shared traversal runs at widths no production
 * backend chooses. The arithmetic stays the portable backend's, since the scheduling is what is under test.
 */
private class RegroupedPanels(private val group: Int) : EngineKernels by ScalarEngine {
    override val implementations = setOf(EngineImplementation.Panels)

    override val name: String get() = "regrouped($group)"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = group
}

class Level2SchedulingTest {

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() = withEngines { blas ->
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25, 65, 130)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        dominateStrictPart(t, explicit, n, lower)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val x = DoubleArray(n)
                        ReferenceBlas.gemv(1.0, explicit, xTrue, 0.0, x, transpose)

                        blas.trsv(t, x, 0, 1, lower, transpose, unitDiag)

                        assertClose(
                            xTrue,
                            x,
                            "trsv n=$n lower=$lower t=$transpose unit=$unitDiag",
                            tolerance = 1e-9,
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `trmv matches gemv on the explicit triangle for all flag combos`() = withEngines { blas ->
        val rng = Random(20260915)
        for (n in intArrayOf(1, 2, 7, 16, 65, 130)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val x = randomVector(n, rng)
                        val expected = DoubleArray(n)
                        ReferenceBlas.gemv(1.0, explicit, x, 0.0, expected, transpose)
                        val actual = x.copyOf()

                        blas.trmv(t, actual, 0, 1, lower, transpose, unitDiag)

                        assertClose(expected, actual, "trmv n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `symv matches gemv on the full matrix and reads only the selected triangle`() = withEngines { blas ->
        val rng = Random(20260910)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 2, 7, 16)) {
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        // The unselected triangle holds NaN, so any read of it poisons the result.
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val x = randomVector(n, rng)
                        val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = y0.copyOf()
                        ReferenceBlas.gemv(alpha, full, x, beta, expected)
                        val actual = y0.copyOf()

                        blas.symv(alpha, poisoned, x, 0, 1, beta, actual, 0, 1, lower)

                        assertClose(expected, actual, "symv n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `gemv agrees with the reference at every grouping and orientation`() {
        val rng = Random(20260921)
        for (group in GROUPS) {
            val blas = regrouped(group)
            for (rows in ORDERS) {
                val cols = if (rows == 0) 0 else rows + 2
                val a = randomMatrix(rows, cols, rng)
                for (transpose in booleanArrayOf(false, true)) {
                    val x = randomVector(if (transpose) rows else cols, rng)
                    val y0 = randomVector(if (transpose) cols else rows, rng)
                    for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                        val expected = y0.copyOf()
                        ReferenceBlas.gemv(0.875, a, x, beta, expected, transpose)
                        val actual = y0.copyOf()

                        blas.gemv(0.875, a, x, 0, 1, beta, actual, 0, 1, transpose)

                        assertClose(expected, actual, "gemv group=$group rows=$rows t=$transpose beta=$beta")
                    }
                }
            }
        }
    }

    @Test
    fun `symv agrees with the reference at every grouping and triangle`() {
        val rng = Random(20260922)
        for (group in GROUPS) {
            val blas = regrouped(group)
            for (n in ORDERS) {
                val a = randomMatrix(n, n, rng)
                val x = randomVector(n, rng)
                val y0 = randomVector(n, rng)
                for (lower in booleanArrayOf(true, false)) {
                    val expected = y0.copyOf()
                    ReferenceBlas.symv(0.875, a, x, -0.25, expected, lower)
                    val actual = y0.copyOf()

                    blas.symv(0.875, a, x, 0, 1, -0.25, actual, 0, 1, lower)

                    assertClose(expected, actual, "symv group=$group n=$n lower=$lower")
                }
            }
        }
    }

    @Test
    fun `rank updates agree with the reference at every grouping and triangle`() {
        val rng = Random(20260923)
        for (group in GROUPS) {
            val blas = regrouped(group)
            for (n in ORDERS) {
                val x = randomVector(n, rng)
                val y = randomVector(n, rng)
                val start = randomMatrix(n, n, rng)
                for (lower in booleanArrayOf(true, false)) {
                    val expected = start.copyOf()
                    ReferenceBlas.syr(0.875, DenseVector.wrap(x), expected, lower)
                    val actual = start.copyOf()
                    blas.syr(
                        0.875,
                        x,
                        0,
                        1,
                        actual,
                        lower,
                    )
                    assertClose(expected, actual, "syr group=$group n=$n lower=$lower")

                    val expectedTwo = start.copyOf()
                    ReferenceBlas.syr2(0.875, DenseVector.wrap(x), DenseVector.wrap(y), expectedTwo, lower)
                    val actualTwo = start.copyOf()
                    blas.syr2(
                        0.875,
                        x,
                        0,
                        1,
                        y,
                        0,
                        1,
                        actualTwo,
                        lower,
                    )
                    assertClose(expectedTwo, actualTwo, "syr2 group=$group n=$n lower=$lower")
                }
                val expectedGer = start.copyOf()
                ReferenceBlas.ger(0.875, x, y, expectedGer)
                val actualGer = start.copyOf()
                blas.ger(0.875, x, 0, 1, y, 0, 1, actualGer)
                assertClose(expectedGer, actualGer, "ger group=$group n=$n")
            }
        }
    }

    // A vector with a step reaches the panel without a copy, so the panel has to address its window.
    @Test
    fun `a strided rank update agrees with the reference`() {
        val rng = Random(20260924)
        val n = 9
        val blas = regrouped(3)
        val buffer = randomVector(2 * n + 1, rng)
        for (stride in intArrayOf(2, -2)) {
            val origin = if (stride > 0) 1 else 2 * n - 1
            val view = DenseVector.wrap(buffer, origin, n, stride)
            val dense = DenseVector.of(DoubleArray(n) { view[it] })
            for (lower in booleanArrayOf(true, false)) {
                val start = randomMatrix(n, n, rng)
                val expected = start.copyOf()
                ReferenceBlas.syr(0.875, dense, expected, lower)
                val actual = start.copyOf()

                blas.syr(0.875, view.values, view.offset, view.stride, actual, lower)

                assertClose(expected, actual, "strided syr stride=$stride lower=$lower")
            }
        }
    }

    /** The shared traversal under a backend recommending [group] columns, with the portable arithmetic. */
    private fun regrouped(group: Int): KoblasEngine = engineWithKernels(ScalarEngine, RegroupedPanels(group))

    /**
     * Shrinks the strict triangle by the order so the diagonal dominates. A random unit triangle is
     * exponentially ill-conditioned in its order, so a solve past a few dozen steps would test the matrix
     * rather than the routine.
     */
    private fun dominateStrictPart(t: DenseMatrix, explicit: DenseMatrix, n: Int, lower: Boolean) {
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (if (lower) i > j else i < j) {
                    t[i, j] /= n
                    explicit[i, j] /= n
                }
            }
        }
    }

    private companion object {
        /**
         * Groupings at every width a traversal could get wrong: one column, a pair, odd widths either side of
         * the production groups, and one wider than any order below.
         */
        val GROUPS = intArrayOf(1, 2, 3, 5, 7, 64)

        val ORDERS = intArrayOf(0, 1, 2, 3, 5, 8, 9, 17)
    }
}
