package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class SymmetricBlasTest {

    @Test
    fun `symv matches gemv on the full matrix and reads only the selected triangle`() = withDenseBlas { blas ->
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

                        blas.symv(alpha, poisoned, x, beta, actual, lower)

                        assertClose(expected, actual, "symv n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    /**
     * A square operand is the whole of what a symmetric routine can mean, so the rejection is the contract.
     *
     * The shape is checked before anything reaches the library, which is why this runs without one.
     */
    @Test
    fun `symv refuses a non square matrix at every size`() {
        for (n in intArrayOf(2, 17, 64, 129)) {
            val wide = DenseMatrix.zero(n, n + 1)
            val tall = DenseMatrix.zero(n + 1, n)
            assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
                koblas.symv(1.0, wide, DoubleArray(n), 0.0, DoubleArray(n))
            }
            assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
                koblas.symv(1.0, tall, DoubleArray(n + 1), 0.0, DoubleArray(n + 1))
            }
        }
    }

    @Test
    fun `symm matches gemm on the full matrix and reads only the selected triangle`() = withDenseBlas { blas ->
        val rng = Random(20260911)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 5, 12)) {
                val p = 4
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val b = randomMatrix(n, p, rng)
                        val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = DenseMatrix(n, p, c0.copyOf())
                        ReferenceBlas.gemm(alpha, full, false, b, false, beta, expected)
                        val actual = DenseMatrix(n, p, c0.copyOf())

                        blas.symm(alpha, poisoned, b, beta, actual, lower)

                        assertClose(expected, actual, "symm n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symm handles empty shapes`() = withDenseBlas { blas ->
        blas.symm(1.0, DenseMatrix(0, 0), DenseMatrix(0, 3), 0.0, DenseMatrix(0, 3))
        blas.symm(1.0, DenseMatrix(2, 2), DenseMatrix(2, 0), 0.0, DenseMatrix(2, 0))
    }

    @Test
    fun `symm leaves C alone when the operands do not line up`() {
        // Scaling C is part of the operation, so a call that cannot go through must not have done it.
        for (right in booleanArrayOf(true, false)) {
            for (beta in doubleArrayOf(0.0, 2.0)) {
                // C matches B, so the shape that fails is A against B: too few columns from the right,
                // too few rows from the left.
                val a = DenseMatrix.diagonal(if (right) 2 else 3)
                val b = DenseMatrix(2, 3)
                val c = DenseMatrix(2, 3)
                c.values.fill(7.0)
                assertFailsWith<DimensionMismatch>("symm right=$right beta=$beta should reject these shapes") {
                    koblas.symm(1.0, a, b, beta, c, lower = true, right = right)
                }
                assertTrue(
                    c.values.all { it == 7.0 },
                    "symm right=$right beta=$beta scaled C before rejecting the call: ${c.values.toList()}",
                )
            }
        }
    }

    @Test
    fun `syrk triangle modes write only the selected triangle with strict beta semantics`() = withDenseBlas { blas ->
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
        blas: DenseBlas,
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
        // The output is NaN where beta == 0 must overwrite without reading, and the unselected triangle stays NaN.
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

    @Test
    fun `the rank two update snapshots a destination sharing either input buffer`() = withDenseBlas { blas ->
        val initial = DenseMatrix.ofRows(Array(3) { i -> DoubleArray(3) { j -> (i * 3 + j + 1).toDouble() / 7.0 } })
        val other = DenseMatrix.wrap(3, 3, initial.values.copyOf()).also {
            for (i in it.values.indices) it.values[i] += 0.25
        }
        for (aliasLeft in booleanArrayOf(true, false)) {
            val expected = DenseMatrix.wrap(3, 3, initial.values.copyOf())
            val left = if (aliasLeft) initial else other
            val right = if (aliasLeft) other else initial
            ReferenceBlas.syr2k(1.0, left, right, false, 0.0, expected)
            val actual = DenseMatrix.wrap(3, 3, initial.values.copyOf())

            blas.syr2k(1.0, if (aliasLeft) actual else other, if (aliasLeft) other else actual, false, 0.0, actual)

            assertClose(expected.values, actual.values, "aliased syr2k left=$aliasLeft")
        }
    }
}
