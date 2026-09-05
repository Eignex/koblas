package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

class TriangularTest {

    @Test
    fun `the triangular block width fits the zero-pivot mask`() {
        // Retuning REFERENCE_TRIANGULAR_BLOCK above 64 wraps the mask shift instead of failing, and the
        // underflow tests below would still pass at n = 65 while being wrong at larger n.
        requireTriangularBlockFitsMask()
    }

    /** `op(explicit) x` by definition, so the solve checks do not depend on the library's own kernels. */
    private fun naiveMultiply(explicit: F64DenseMatrix, transpose: Boolean, x: DoubleArray): DoubleArray {
        val n = explicit.rows
        val y = DoubleArray(n)
        for (i in 0 until n) {
            for (j in 0 until n) y[i] += (if (transpose) explicit[j, i] else explicit[i, j]) * x[j]
        }
        return y
    }

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() {
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val x = naiveMultiply(explicit, transpose, xTrue)
                        t.trsv(x, lower, transpose, unitDiag)
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
    fun `trsm matches per-column trsv across the flag combinations`() {
        val rng = Random(20260729)
        val n = 12
        val nrhs = 5
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(n, nrhs, rng)
                    val viaTrsm = F64DenseMatrix(n, nrhs, b.data.copyOf())
                    t.trsm(viaTrsm, lower, transpose, unitDiag, workspace = Workspace())
                    for (c in 0 until nrhs) {
                        val col = DoubleArray(n) { b[it, c] }
                        t.trsv(col, lower, transpose, unitDiag)
                        for (i in 0 until n) {
                            assertTrue(
                                abs(viaTrsm[i, c] - col[i]) < 1e-12,
                                "trsm lower=$lower t=$transpose unit=$unitDiag col $c row $i",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `trsm solves the multi-RHS system against a gemm residual check`() {
        val rng = Random(20260730)
        val n = 10
        val nrhs = 3
        val (t, explicit) = poisonedTriangle(rng, n, lower = true, unitDiag = false)
        val xTrue = randomMatrix(n, nrhs, rng)
        val b = explicit * xTrue
        t.trsm(b, lower = true)
        assertClose(xTrue, b, "trsm multi-RHS", tolerance = 1e-9)
    }

    @Test
    fun `trsm applies alpha to the solved matrix`() {
        val rng = Random(20261021)
        val alpha = -0.75
        val n = 10
        val nrhs = 3
        val (t, explicit) = poisonedTriangle(rng, n, lower = true, unitDiag = false)
        val x = randomMatrix(n, nrhs, rng)
        val b = explicit * x

        t.trsm(b, lower = true, alpha = alpha)

        val expected = F64DenseMatrix(n, nrhs, DoubleArray(n * nrhs) { alpha * x.data[it] })
        assertClose(expected, b, "trsm alpha", tolerance = 1e-9)
    }

    @Test
    fun `trmv matches gemv on the explicit triangle for all flag combos`() {
        val rng = Random(20260915)
        for (n in intArrayOf(1, 2, 7, 16)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val x = randomVector(n, rng)
                        val expected = koblas.gemv(explicit, x, transpose)
                        val actual = x.copyOf()
                        t.trmv(actual, lower, transpose, unitDiag)
                        assertClose(expected, actual, "trmv n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm matches gemm on the explicit triangle for all flag combos`() {
        val rng = Random(20260916)
        val alpha = -0.75
        for (n in intArrayOf(1, 5, 12)) {
            val p = 3
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val b = randomMatrix(n, p, rng)
                        val expected = F64DenseMatrix(n, p)
                        koblas.gemm(alpha, explicit, transpose, b, false, 0.0, expected)
                        val actual = F64DenseMatrix(n, p, b.data.copyOf())
                        t.trmm(actual, lower, transpose, unitDiag, alpha = alpha)
                        assertClose(expected, actual, "trmm n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm right multiplies from the right and matches gemm`() {
        val rng = Random(20260940)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(rows, n, rng)
                    val expected = F64DenseMatrix(rows, n)
                    koblas.gemm(1.0, b, false, explicit, transpose, 0.0, expected)
                    val actual = F64DenseMatrix.wrap(rows, n, b.data.copyOf())
                    t.trmm(actual, lower, transpose, unitDiag, right = true)
                    assertClose(expected, actual, "trmm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `trsm right inverts trmm right`() {
        val rng = Random(20260941)
        val n = 6
        val rows = 3
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val x = randomMatrix(rows, n, rng)
                    val b = F64DenseMatrix.wrap(rows, n, x.data.copyOf())
                    t.trmm(b, lower, transpose, unitDiag, right = true)
                    t.trsm(b, lower, transpose, unitDiag, right = true, workspace = Workspace())
                    assertClose(x, b, "trsm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `triangular matrix operations cross the diagonal block boundary`() {
        val rng = Random(20261031)
        val n = REFERENCE_TRIANGULAR_BLOCK + 5
        val width = 3
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unitDiag in booleanArrayOf(false, true)) {
                    val (triangle, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                    val original = randomMatrix(n, width, rng)
                    val multiplied = F64DenseMatrix(n, width, original.data.copyOf())
                    triangle.trmm(multiplied, lower, transpose, unitDiag)
                    for (column in 0 until width) {
                        val expected = DoubleArray(n) { original[it, column] }
                        triangle.trmv(expected, lower, transpose, unitDiag)
                        assertClose(expected, DoubleArray(n) { multiplied[it, column] }, "left multiply")
                    }
                    triangle.trsm(multiplied, lower, transpose, unitDiag)
                    assertClose(original, multiplied, "left round trip", tolerance = 1e-9)

                    val rightOriginal = randomMatrix(width, n, rng)
                    val right = F64DenseMatrix(width, n, rightOriginal.data.copyOf())
                    triangle.trmm(right, lower, transpose, unitDiag, right = true)
                    val expectedRight = F64DenseMatrix(width, n)
                    for (j in 0 until n) {
                        for (i in 0 until width) {
                            for (p in 0 until n) {
                                expectedRight[i, j] += rightOriginal[i, p] *
                                    (if (transpose) explicit[j, p] else explicit[p, j])
                            }
                        }
                    }
                    assertClose(expectedRight, right, "right multiply", tolerance = 1e-10)
                    triangle.trsm(right, lower, transpose, unitDiag, right = true)
                    assertClose(rightOriginal, right, "right round trip", tolerance = 1e-9)
                }
            }
        }
    }

    @Test
    fun `trmv round-trips with trsv`() {
        val rng = Random(20260917)
        val n = 9
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val (t, _) = poisonedTriangle(rng, n, lower, unitDiag = false)
                val x0 = randomVector(n, rng)
                val x = x0.copyOf()
                t.trmv(x, lower, transpose)
                t.trsv(x, lower, transpose)
                assertClose(x0, x, "roundtrip lower=$lower t=$transpose")
            }
        }
    }

    @Test
    fun `trtri inverts a triangle in both orientations`() {
        val rng = Random(20260806)
        for (n in intArrayOf(1, 4, 7)) {
            for (lower in booleanArrayOf(true, false)) {
                val t = F64DenseMatrix(n, n)
                for (j in 0 until n) {
                    for (i in 0 until n) {
                        val inTriangle = if (lower) i >= j else i <= j
                        if (inTriangle) t[i, j] = if (i == j) 2.0 + j else rng.nextDouble(-1.0, 1.0)
                    }
                }
                val inv = t.trtri(lower)
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        var s = 0.0
                        for (k in 0 until n) s += t[i, k] * inv[k, j]
                        assertClose(if (i == j) 1.0 else 0.0, s, "n=$n lower=$lower at [$i,$j]", tolerance = 1e-9)
                    }
                    for (j in 0 until n) {
                        val outside = if (lower) i < j else i > j
                        if (outside) {
                            assertClose(0.0, inv[i, j], "n=$n lower=$lower leaked at [$i,$j]", tolerance = 0.0)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `trtri rejects a zero on the diagonal`() {
        val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(3.0, 0.0)))
        val failure = assertFailsWith<SingularMatrix> { singular.trtri(lower = true) }
        assertTrue("entry 1" in failure.message!!, "should name the zero position: ${failure.message}")
        singular.trtri(lower = true, unitDiag = true)
    }

    @Test
    fun `trsv divides by a zero diagonal instead of reporting it`() {
        val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(3.0, 0.0)))
        val x = doubleArrayOf(1.0, 1.0)
        F64ReferenceLinearAlgebra.trsv(singular, x, lower = true)
        assertTrue(!x[1].isFinite(), "expected a non-finite entry from the zero pivot, got ${x[1]}")
    }

    @Test
    fun `nontransposed trsv skips a zero right hand side pivot`() {
        val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(Double.NaN, 1.0)))
        val x = DoubleArray(2)

        F64ReferenceLinearAlgebra.trsv(singular, x, lower = true)

        assertContentEquals(DoubleArray(2), x)
    }

    @Test
    fun `transposed trsv divides a zero right hand side pivot`() {
        val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 1.0)))
        val x = DoubleArray(2)

        F64ReferenceLinearAlgebra.trsv(singular, x, lower = true, transpose = true)

        assertTrue(x[0].isNaN())
    }

    @Test
    fun `transposed trsv forms products with zero triangle entries`() {
        val triangle = F64DenseMatrix.diagonal(2)
        val x = doubleArrayOf(0.0, Double.POSITIVE_INFINITY)

        F64ReferenceLinearAlgebra.trsv(triangle, x, lower = true, transpose = true)

        assertTrue(x[0].isNaN())
    }

    @Test
    fun `right trsm skips zero triangle coefficients`() {
        val triangle = F64DenseMatrix.diagonal(2)
        val b = F64DenseMatrix(1, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0))

        F64ReferenceLinearAlgebra.trsm(triangle, b, lower = true, right = true)

        assertEquals(Double.POSITIVE_INFINITY, b[0, 0])
        assertEquals(1.0, b[0, 1])
    }

    @Test
    fun `right trmm skips zero triangle coefficients`() {
        val triangle = F64DenseMatrix.diagonal(2)
        val b = F64DenseMatrix(1, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0))

        F64ReferenceLinearAlgebra.trmm(triangle, b, lower = true, right = true)

        assertEquals(Double.POSITIVE_INFINITY, b[0, 0])
        assertEquals(1.0, b[0, 1])
    }

    @Test
    fun `blocked trsm retains products after a quotient underflows`() {
        val n = REFERENCE_TRIANGULAR_BLOCK + 1
        val triangle = F64DenseMatrix.diagonal(n)
        triangle[0, 0] = Double.POSITIVE_INFINITY
        triangle[n - 1, 0] = Double.POSITIVE_INFINITY
        val b = F64DenseMatrix(n, 1).also { it[0, 0] = Double.MIN_VALUE }

        F64ReferenceLinearAlgebra.trsm(triangle, b, lower = true)

        assertTrue(b[n - 1, 0].isNaN(), "underflowed pivot did not form the cross-block product")
    }

    @Test
    fun `trsv and trsm validate shapes`() {
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(2, 3).trsv(DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(3, 3).trsv(DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { F64DenseMatrix(3, 3).trsm(F64DenseMatrix(2, 4), lower = true) }
    }
}
