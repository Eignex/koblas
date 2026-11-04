package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.PortableEngine
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A call with no products to form, either because its multiplier is zero or because its depth is, scales the
 * destination and reads neither operand nor, over an empty depth, the multiplier. Both are poisoned here, and
 * the oracle is handed zero operands and a zero multiplier, which leave it the same beta term.
 */
class Level3OpsTest {
    @Test
    fun `no work products scale the destination without reading operands or multiplier`() =
        withProductKernels { products ->
            val blas = engineWithKernels(PortableEngine, PortableEngine, products)
            val rng = Random(20261014)
            for ((alpha, depth) in NO_WORK) {
                for (transposeA in booleanArrayOf(false, true)) {
                    for (transposeB in booleanArrayOf(false, true)) {
                        val a = poisoned(if (transposeA) depth else 4, if (transposeA) 4 else depth)
                        val b = poisoned(if (transposeB) 5 else depth, if (transposeB) depth else 5)
                        for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                            val start = randomMatrix(4, 5, rng)
                            if (beta == 0.0) start.values.fill(Double.NaN)
                            val expected = start.copyOf()
                            ReferenceBlas.gemm(0.0, zerosLike(a), transposeA, zerosLike(b), transposeB, beta, expected)
                            val actual = start.copyOf()

                            blas.gemm(alpha, a, transposeA, b, transposeB, beta, actual)

                            val context = "${products.name} alpha=$alpha depth=$depth " +
                                "tA=$transposeA tB=$transposeB beta=$beta"
                            assertClose(expected, actual, context)
                            if (beta == 1.0) assertContentEquals(start.values, actual.values, context)
                        }
                    }
                }
            }
        }

    @Test
    fun `no work triangle products scale the selected triangle and preserve the other`() = withEngines { blas ->
        val rng = Random(20261015)
        for ((alpha, depth) in NO_WORK) {
            for (transposeA in booleanArrayOf(false, true)) {
                val a = poisoned(if (transposeA) depth else 4, if (transposeA) 4 else depth)
                val other = poisoned(a.rows, a.cols)
                val zeros = zerosLike(a)
                for (lower in booleanArrayOf(false, true)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.25)) {
                        val context = "${blas.name} alpha=$alpha depth=$depth tA=$transposeA"
                        for (transposeB in booleanArrayOf(false, true)) {
                            val b = poisoned(if (transposeB) 4 else depth, if (transposeB) depth else 4)
                            val zeroB = zerosLike(b)
                            assertNoWorkTriangle("gemmt $context tB=$transposeB", rng, lower, beta, { expected ->
                                ReferenceBlas.gemmt(0.0, zeros, transposeA, zeroB, transposeB, beta, expected, lower)
                            }) { actual -> blas.gemmt(alpha, a, transposeA, b, transposeB, beta, actual, lower) }
                        }
                        assertNoWorkTriangle("syrk $context", rng, lower, beta, { expected ->
                            ReferenceBlas.syrk(0.0, zeros, transposeA, beta, expected, lower)
                        }) { actual -> blas.syrk(alpha, a, transposeA, beta, actual, lower) }
                        assertNoWorkTriangle("syr2k $context", rng, lower, beta, { expected ->
                            ReferenceBlas.syr2k(0.0, zeros, zeros, transposeA, beta, expected, lower)
                        }) { actual -> blas.syr2k(alpha, a, other, transposeA, beta, actual, lower) }
                    }
                }
            }
        }
    }

    @Suppress("LongParameterList") // the operation, its triangle and scaling, and the two calls compared
    private fun assertNoWorkTriangle(
        operation: String,
        rng: Random,
        lower: Boolean,
        beta: Double,
        reference: (DenseMatrix) -> Unit,
        call: (DenseMatrix) -> Unit,
    ) {
        val start = randomMatrix(4, 4, rng)
        if (beta == 0.0) {
            for (j in 0 until 4) for (i in 0 until 4) if (if (lower) i >= j else i <= j) start[i, j] = Double.NaN
        }
        val expected = start.copyOf()
        val actual = start.copyOf()
        reference(expected)

        call(actual)

        val context = "$operation beta=$beta lower=$lower"
        assertClose(expected, actual, context)
        for (j in 0 until 4) {
            for (i in 0 until 4) {
                if (beta == 1.0 || (if (lower) i < j else i > j)) {
                    assertEquals(start[i, j], actual[i, j], "$context at ($i, $j)")
                }
            }
        }
    }

    @Test
    fun `syrk triangle modes write only the selected triangle with strict beta semantics`() = withEngines { blas ->
        val rng = Random(20260933)
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        checkSyrk(blas, rng, lower, transpose, alpha, beta)
                    }
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod", "LongParameterList") // one loop variable per flag the routine carries
    private fun checkSyrk(
        blas: KoblasEngine,
        rng: Random,
        lower: Boolean,
        transpose: Boolean,
        alpha: Double,
        beta: Double,
    ) {
        val a = randomMatrix(6, 4, rng)
        val n = if (transpose) a.cols else a.rows
        val term = DenseMatrix(n, n)
        ReferenceBlas.syrk(1.0, a, transpose, 0.0, term, lower)
        // The output is NaN where a zero beta must overwrite without reading, and the unselected triangle
        // stays NaN.
        val c = DenseMatrix(n, n)
        val c0 = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j <= i else j >= i
                val v = if (!selected || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                c[i, j] = v
                c0[i, j] = v
            }
        }

        blas.syrk(alpha, a, transpose, beta, c, lower)

        val context = "syrk lower=$lower t=$transpose a=$alpha b=$beta"
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (lower) j <= i else j >= i
                if (selected) {
                    val expected = (if (beta == 0.0) 0.0 else beta * c0[i, j]) + alpha * term[i, j]
                    assertClose(expected, c[i, j], "$context ($i;$j)")
                } else {
                    assertTrue(c[i, j].isNaN(), "$context ($i;$j): untouched triangle was written")
                }
            }
        }
    }

    /** A matrix of [rows] by [cols] whose every entry is NaN, so any read of it shows in the result. */
    private fun poisoned(rows: Int, cols: Int): DenseMatrix =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { Double.NaN })

    private fun zerosLike(a: DenseMatrix): DenseMatrix = DenseMatrix.zero(a.rows, a.cols)

    private companion object {
        /** A zero multiplier over a real depth, and an unreadable multiplier over no depth at all. */
        val NO_WORK = listOf(0.0 to 3, Double.NaN to 0)
    }
}
