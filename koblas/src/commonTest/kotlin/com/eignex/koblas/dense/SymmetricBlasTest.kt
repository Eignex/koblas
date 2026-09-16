package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class SymmetricBlasTest {

    /** The `dsyr2k` sum for one entry, over whichever orientation [transpose] selects. */
    private fun syr2kEntry(a: DenseMatrix, b: DenseMatrix, transpose: Boolean, k: Int, i: Int, j: Int): Double {
        var s = 0.0
        for (p in 0 until k) {
            val ai = if (transpose) a[p, i] else a[i, p]
            val aj = if (transpose) a[p, j] else a[j, p]
            val bi = if (transpose) b[p, i] else b[i, p]
            val bj = if (transpose) b[p, j] else b[j, p]
            s += ai * bj + bi * aj
        }
        return s
    }

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

    @Test
    fun `symv with zero alpha overwrites the destination`() = withDenseBlas { blas ->
        val y = DoubleArray(2) { Double.NaN }

        blas.symv(0.0, DenseMatrix(2, 2), DoubleArray(2), 0.0, y)

        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }

    /**
     * A square operand is the whole of what a symmetric routine can mean, so the rejection is the contract.
     *
     * The shape is checked before anything reaches the library, which is why this runs without one.
     */
    @Test
    fun `symv refuses a non-square matrix at every size`() {
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
    fun `symm right multiplies from the right and reads only the selected triangle`() = withDenseBlas { blas ->
        val rng = Random(20260942)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (alpha in doubleArrayOf(0.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                    val b = randomMatrix(rows, n, rng)
                    val c0 = DoubleArray(rows * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = DenseMatrix.wrap(rows, n, c0.copyOf())
                    ReferenceBlas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = DenseMatrix.wrap(rows, n, c0.copyOf())

                    blas.symm(alpha, poisoned, b, beta, actual, lower, right = true)

                    assertClose(expected, actual, "symm right l=$lower a=$alpha b=$beta", tolerance = 1e-11)
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
                c.data.fill(7.0)
                assertFailsWith<DimensionMismatch>("symm right=$right beta=$beta should reject these shapes") {
                    koblas.symm(1.0, a, b, beta, c, lower = true, right = right)
                }
                assertTrue(
                    c.data.all { it == 7.0 },
                    "symm right=$right beta=$beta scaled C before rejecting the call: ${c.data.toList()}",
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

    /**
     * A triangle mode has to beta-scale and write the same half, and leave the other half of an asymmetric
     * destination exactly as it found it. Beta is non-zero and C starts asymmetric so that scaling the wrong
     * region, or mirroring where the routine should accumulate, both show up.
     */
    @Test
    fun `syr2k writes only the triangle it is given and scales only that`() = withDenseBlas { blas ->
        val rng = Random(20260825)
        val n = 5
        val k = 3
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(true, false)) {
                val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
                val before = randomMatrix(n, n, rng)
                val c = DenseMatrix.wrap(n, n, before.data.copyOf())

                blas.syr2k(0.5, a, b, transpose, beta = 2.0, c = c, lower = lower)

                for (i in 0 until n) {
                    for (j in 0 until n) {
                        val written = if (lower) i >= j else i <= j
                        val expected = if (written) {
                            2.0 * before[i, j] + 0.5 * syr2kEntry(a, b, transpose, k, i, j)
                        } else {
                            before[i, j]
                        }
                        assertClose(expected, c[i, j], "transpose=$transpose lower=$lower at [$i,$j]")
                    }
                }
            }
        }
    }

    @Test
    fun `syr2k matches the gemm expansion in both orientations`() = withDenseBlas { blas ->
        val rng = Random(20260809)
        for (transpose in booleanArrayOf(false, true)) {
            val n = 4
            val k = 3
            val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val c = DenseMatrix(n, n)

            blas.syr2k(0.5, a, b, transpose, 0.0, c)

            for (j in 0 until n) {
                for (i in j until n) {
                    assertClose(
                        0.5 * syr2kEntry(a, b, transpose, k, i, j),
                        c[i, j],
                        "transpose=$transpose at [$i,$j]",
                    )
                }
            }
        }
    }

    /**
     * BLAS leaves a destination overlapping an input undefined, and nothing here copies one out of the way.
     *
     * The rank updates used to stage an aliased operand and keep working. They cannot now: the operands go
     * straight to the library, so the only honest answer is to refuse the call rather than to hand the vendor
     * a case its own contract does not cover and report whatever comes back.
     */
    @Test
    fun `the rank updates refuse a destination that shares a buffer with an input`() {
        val shared = DenseMatrix.wrap(3, 3, DoubleArray(9))
        val other = DenseMatrix(3, 3)

        assertFailsWith<IllegalArgumentException> { koblas.syrk(1.0, shared, false, 0.0, shared) }
        assertFailsWith<IllegalArgumentException> { koblas.syr2k(1.0, shared, other, false, 0.0, shared) }
        assertFailsWith<IllegalArgumentException> { koblas.syr2k(1.0, other, shared, false, 0.0, shared) }
    }
}
