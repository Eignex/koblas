@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.StridedVector
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * A backend that recommends a grouping of its own, so the shared traversal runs at widths no production
 * backend chooses. The arithmetic stays the portable backend's, since the scheduling is what is under test.
 */
private class RegroupedPanels(private val group: Int) : KoblasEngine() {
    override val implementations = setOf(EngineImplementation.Panels)

    override val name: String get() = "regrouped($group)"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = group
}

/**
 * The shared dense scheduling, at every grouping a backend might recommend, since a traversal right at four
 * columns and wrong at three would pass on this machine and fail on the next one.
 */
class PortableDenseBlasTest {
    @Test
    fun `GER agrees with the reference on shrinking supports and extreme finite inputs`() {
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()) {
            assertShrinkingGerAgreesWithReference { x, y, a -> engine.ger(-1.0, x, y, a) }
        }
    }

    @Test
    fun `product aliases agree with the reference and share one snapshot`() {
        withDenseBlas { blas -> assertDenseProductAliasesAgreeWithReference(blas) }
    }

    @Test
    fun `rectangular product aliases retain their operand shapes`() {
        withDenseBlas { blas -> assertRectangularProductAliasesAgreeWithReference(blas) }
    }

    private val groups = intArrayOf(1, 2, 3, 5, 7, 64)

    private fun blasFor(group: Int): KoblasEngine = engineWithKernels(PortableEngine, RegroupedPanels(group))

    private fun orders() = intArrayOf(0, 1, 2, 3, 5, 8, 9, 17)

    @Test
    fun `no work products agree with the reference for dense operands`() = withProducts { products ->
        val blas = engineWithKernels(PortableEngine, PortablePanelKernels, products)
        val rng = Random(20261014)
        for ((alpha, depth) in listOf(0.0 to 3, 0.875 to 0)) {
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    val a = randomMatrix(if (transposeA) depth else 4, if (transposeA) 4 else depth, rng)
                    val b = randomMatrix(if (transposeB) 5 else depth, if (transposeB) depth else 5, rng)
                    for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                        val start = randomMatrix(4, 5, rng)
                        if (beta == 0.0) start.values.fill(Double.NaN)
                        val expected = start.copyOf()
                        ReferenceBlas.gemm(alpha, a, transposeA, b, transposeB, beta, expected)
                        val calls = listOf<(DenseMatrix) -> Unit>(
                            { blas.gemm(alpha, a, transposeA, b, transposeB, beta, it) },
                        )

                        for ((variant, call) in calls.withIndex()) {
                            val actual = start.copyOf()
                            call(actual)

                            val context = "${products.name} variant=$variant alpha=$alpha depth=$depth " +
                                "tA=$transposeA tB=$transposeB beta=$beta"
                            assertClose(expected, actual, context)
                            if (beta == 1.0) assertContentEquals(start.values, actual.values, context)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `no work triangle products agree with the reference and preserve the other triangle`() {
        val blas = blasFor(3)
        val rng = Random(20261015)
        for ((alpha, depth) in listOf(0.0 to 3, 0.875 to 0)) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = randomMatrix(if (transpose) depth else 4, if (transpose) 4 else depth, rng)
                val b = randomMatrix(a.rows, a.cols, rng)
                for (lower in booleanArrayOf(false, true)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                        for (operation in listOf("gemmt", "syrk", "syr2k")) {
                            val start = randomMatrix(4, 4, rng)
                            if (beta == 0.0) {
                                for (j in 0 until 4) {
                                    for (i in 0 until 4) {
                                        if (if (lower) i >= j else i <= j) start[i, j] = Double.NaN
                                    }
                                }
                            }
                            val expected = start.copyOf()
                            val actual = start.copyOf()

                            when (operation) {
                                "gemmt" -> {
                                    ReferenceBlas.gemmt(alpha, a, transpose, b, !transpose, beta, expected, lower)
                                    blas.gemmt(alpha, a, transpose, b, !transpose, beta, actual, lower)
                                }

                                "syrk" -> {
                                    ReferenceBlas.syrk(alpha, a, transpose, beta, expected, lower)
                                    blas.syrk(alpha, a, transpose, beta, actual, lower)
                                }

                                "syr2k" -> {
                                    ReferenceBlas.syr2k(alpha, a, b, transpose, beta, expected, lower)
                                    blas.syr2k(alpha, a, b, transpose, beta, actual, lower)
                                }
                            }

                            val context = "$operation alpha=$alpha depth=$depth t=$transpose beta=$beta lower=$lower"
                            assertClose(expected, actual, context)
                            for (j in 0 until 4) {
                                for (i in 0 until 4) {
                                    if (beta == 1.0 || (if (lower) i < j else i > j)) {
                                        assertEquals(start[i, j], actual[i, j], "$context at ($i, $j)")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `no work products validate shapes before returning`() {
        val blas = blasFor(3)
        for ((alpha, depth) in listOf(0.0 to 3, 0.875 to 0)) {
            val a = DenseMatrix(4, depth)
            val b = DenseMatrix(depth, 4)
            val c = DenseMatrix(4, 4, DoubleArray(16) { 7.0 })
            val wrongOutput = DenseMatrix(4, 3, DoubleArray(12) { 7.0 })
            val calls = listOf<() -> Unit>(
                { blas.gemm(alpha, a, false, DenseMatrix(depth + 1, 4), false, 1.0, c) },
                { blas.gemm(alpha, a, false, b, false, 1.0, wrongOutput) },
                { blas.gemmt(alpha, a, false, DenseMatrix(depth + 1, 4), false, 1.0, c) },
                { blas.gemmt(alpha, a, false, b, false, 1.0, wrongOutput) },
                { blas.syrk(alpha, DenseMatrix(3, depth), false, 1.0, c) },
                { blas.syrk(alpha, a, false, 1.0, wrongOutput) },
                { blas.syr2k(alpha, a, DenseMatrix(4, depth + 1), false, 1.0, c) },
                { blas.syr2k(alpha, a, a, false, 1.0, wrongOutput) },
            )

            for (call in calls) assertFailsWith<DimensionMismatch> { call() }

            assertContentEquals(DoubleArray(16) { 7.0 }, c.values)
            assertContentEquals(DoubleArray(12) { 7.0 }, wrongOutput.values)
        }
    }

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
                    val expected = start.copyOf()
                    ReferenceBlas.syr(0.875, DenseVector.wrap(x), expected, lower)
                    val actual = start.copyOf()
                    blas.syr(0.875, DenseVector.wrap(x), actual, lower)
                    assertClose(expected, actual, "syr group=$group n=$n lower=$lower")

                    val expectedTwo = start.copyOf()
                    ReferenceBlas.syr2(0.875, DenseVector.wrap(x), DenseVector.wrap(y), expectedTwo, lower)
                    val actualTwo = start.copyOf()
                    blas.syr2(0.875, DenseVector.wrap(x), DenseVector.wrap(y), actualTwo, lower)
                    assertClose(expectedTwo, actualTwo, "syr2 group=$group n=$n lower=$lower")
                }
                val expectedGer = start.copyOf()
                ReferenceBlas.ger(0.875, x, y, expectedGer)
                val actualGer = start.copyOf()
                blas.ger(0.875, x, y, actualGer)
                assertClose(expectedGer, actualGer, "ger group=$group n=$n")
            }
        }
    }

    // A vector with a step reaches the panel without a copy, so the panel has to address its window.
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
                val expected = start.copyOf()
                ReferenceBlas.syr(0.875, dense, expected, lower)
                val actual = start.copyOf()
                blas.syr(0.875, view, actual, lower)
                assertClose(expected, actual, "strided syr stride=$stride lower=$lower")
            }
        }
    }
}
