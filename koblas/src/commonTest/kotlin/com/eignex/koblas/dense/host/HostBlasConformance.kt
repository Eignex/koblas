package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

private val reference = F64ReferenceBlas
internal fun assertLevel3AgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260729)
    for (n in sizes) {
        val a = randomMatrix(n, n, rng)
        val b = randomMatrix(n, n, rng)
        assertClose(reference.gemm(a, b), blas.gemm(a, b), "gemm n=$n", tolerance = 1e-9 * n)
        for (lower in booleanArrayOf(true, false)) {
            val expected = F64DenseMatrix(n, n)
            val actual = F64DenseMatrix(n, n)
            reference.syrk(1.0, a, transpose = true, beta = 0.0, c = expected, lower = lower)
            blas.syrk(1.0, a, transpose = true, beta = 0.0, c = actual, lower = lower)
            assertClose(expected, actual, "syrk lower=$lower n=$n", tolerance = 1e-9 * n)
        }
    }
}

internal fun assertGerAgreesWithReference(blas: F64Blas) {
    val rng = Random(20260802)
    for ((rows, cols) in listOf(1 to 1, 4 to 7, 9 to 3, 30 to 9)) {
        for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
            val x = DoubleArray(rows) { rng.nextDouble(-1.0, 1.0) }
            val y = DoubleArray(cols) { rng.nextDouble(-1.0, 1.0) }
            val a0 = randomMatrix(rows, cols, rng)
            val expected = F64DenseMatrix(rows, cols, a0.data.copyOf())
            reference.ger(alpha, x, y, expected)
            val actual = F64DenseMatrix(rows, cols, a0.data.copyOf())
            blas.ger(alpha, x, y, actual)
            assertClose(expected, actual, "ger ${rows}x$cols alpha=$alpha")
        }
    }
}

internal fun assertSyrAgreesWithReference(blas: F64Blas) {
    val rng = Random(20260831)
    for (n in intArrayOf(1, 7, 31)) {
        for (lower in booleanArrayOf(true, false)) {
            val x = F64DenseVector.of(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
            val y = F64DenseVector.of(DoubleArray(n) { rng.nextDouble(-1.0, 1.0) })
            val seed = randomMatrix(n, n, rng)
            val syrExpected = F64DenseMatrix(n, n, seed.data.copyOf())
            val syrActual = F64DenseMatrix(n, n, seed.data.copyOf())
            reference.syr(0.75, x, syrExpected, lower)
            blas.syr(0.75, x, syrActual, lower)
            assertClose(syrExpected, syrActual, "syr n=$n lower=$lower")
            val syr2Expected = F64DenseMatrix(n, n, seed.data.copyOf())
            val syr2Actual = F64DenseMatrix(n, n, seed.data.copyOf())
            reference.syr2(-0.5, x, y, syr2Expected, lower)
            blas.syr2(-0.5, x, y, syr2Actual, lower)
            assertClose(syr2Expected, syr2Actual, "syr2 n=$n lower=$lower")
        }
    }
}

/** The unselected triangle is NaN, so reading outside the promised one fails the comparison. */
internal fun assertTriangularAgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260801)
    for (n in sizes) {
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    checkTriangular(blas, rng, n, lower, transpose, unitDiag)
                }
            }
        }
    }
}

@Suppress("LongParameterList") // the flag combination under test
private fun checkTriangular(blas: F64Blas, rng: Random, n: Int, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
    val t = F64DenseMatrix(n, n)
    for (i in 0 until n) {
        for (j in 0 until n) {
            val selected = if (lower) j < i else j > i
            t[i, j] = when {
                selected -> rng.nextDouble(-1.0, 1.0)
                i == j -> if (unitDiag) Double.NaN else rng.nextDouble(2.0, 4.0)
                else -> Double.NaN
            }
        }
    }
    val flags = "n=$n lower=$lower t=$transpose unit=$unitDiag"
    val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    for ((name, op) in listOf<Pair<String, (F64Blas, DoubleArray) -> Unit>>(
        "trsv" to { la, v -> la.trsv(t, v, lower, transpose, unitDiag) },
        "trmv" to { la, v -> la.trmv(t, v, lower, transpose, unitDiag) },
    )) {
        val expected = x.copyOf().also { op(reference, it) }
        val actual = x.copyOf().also { op(blas, it) }
        assertClose(expected, actual, "$name $flags", tolerance = 1e-9)
    }
    val nrhs = 3
    for (right in booleanArrayOf(false, true)) {
        val b = if (right) randomMatrix(nrhs, n, rng) else randomMatrix(n, nrhs, rng)
        for ((name, op) in listOf<Pair<String, (F64Blas, F64DenseMatrix) -> Unit>>(
            "trsm" to { la, m -> la.trsm(t, m, lower, transpose, unitDiag, right, alpha = -0.75) },
            "trmm" to { la, m -> la.trmm(t, m, lower, transpose, unitDiag, right, alpha = -0.75) },
        )) {
            val expected = F64DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(reference, it) }
            val actual = F64DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(blas, it) }
            assertClose(expected, actual, "$name right=$right $flags", tolerance = 1e-9)
        }
    }
}

/** The vector solve both ways, then the blocked multi-RHS path where a native trsm earns its call. */
internal fun assertSymvRefusesNonSquare(blas: F64Blas) {
    for (n in intArrayOf(2, 17, 64, 129)) {
        assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
            blas.symv(1.0, F64DenseMatrix.zero(n, n + 1), DoubleArray(n), 0.0, DoubleArray(n))
        }
        assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
            blas.symv(1.0, F64DenseMatrix.zero(n + 1, n), DoubleArray(n + 1), 0.0, DoubleArray(n + 1))
        }
    }
}

/**
 * `dsytrs` divides by a zero pivot and still reports success, so a backend that skips the singularity check
 * returns infinities where it should throw. [nrhs] must reach the width at which the native
 * multi-right-hand-side path takes over.
 */
internal fun assertGemvAgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260727)
    for (size in sizes) checkGemv(blas, rng, rows = size, cols = maxOf(1, size - 2))
}

private fun checkGemv(blas: F64Blas, rng: Random, rows: Int, cols: Int) {
    for (transpose in booleanArrayOf(false, true)) {
        for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
            for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                val a = randomMatrix(rows, cols, rng)
                val x = DoubleArray(if (transpose) rows else cols) { rng.nextDouble(-1.0, 1.0) }
                val yLen = if (transpose) cols else rows
                val y0 = DoubleArray(yLen) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                val expected = y0.copyOf()
                val actual = y0.copyOf()
                reference.gemv(alpha, a, x, beta, expected, transpose)
                blas.gemv(alpha, a, x, beta, actual, transpose)
                assertClose(expected, actual, "gemv ${rows}x$cols t=$transpose a=$alpha b=$beta")
            }
        }
    }
}

/** `C` starts as NaN wherever beta is zero, so a backend that reads it instead of writing it fails. */
internal fun assertGemmAgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260728)
    for (size in sizes) {
        checkGemm(blas, rng, m = size, k = maxOf(1, size - 2), n = maxOf(1, size - 1))
    }
}

@Suppress("LongParameterList") // the three extents of the product
private fun checkGemm(blas: F64Blas, rng: Random, m: Int, k: Int, n: Int) {
    for (transposeA in booleanArrayOf(false, true)) {
        for (transposeB in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val a = if (transposeA) randomMatrix(k, m, rng) else randomMatrix(m, k, rng)
                    val b = if (transposeB) randomMatrix(n, k, rng) else randomMatrix(k, n, rng)
                    val c0 = DoubleArray(m * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = F64DenseMatrix.wrap(m, n, c0.copyOf())
                    val actual = F64DenseMatrix.wrap(m, n, c0.copyOf())
                    reference.gemm(alpha, a, transposeA, b, transposeB, beta, expected)
                    blas.gemm(alpha, a, transposeA, b, transposeB, beta, actual)
                    val ctx = "gemm ${m}x${k}x$n tA=$transposeA tB=$transposeB a=$alpha b=$beta"
                    assertClose(expected.data, actual.data, ctx)
                }
            }
        }
    }
}

/** A symmetric rank update agrees with the portable implementation over its selected triangle. */
internal fun assertSyrkAgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260729)
    for (size in sizes) checkSyrk(blas, rng, rows = size, cols = maxOf(1, size - 2))
}

private fun checkSyrk(blas: F64Blas, rng: Random, rows: Int, cols: Int) {
    for (transpose in booleanArrayOf(false, true)) {
        for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
            for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                val a = randomMatrix(rows, cols, rng)
                val n = if (transpose) cols else rows
                val c0 = DoubleArray(n * n) { index ->
                    val i = index % n
                    val j = index / n
                    if (i < j || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                }
                val expected = F64DenseMatrix.wrap(n, n, c0.copyOf())
                val actual = F64DenseMatrix.wrap(n, n, c0.copyOf())
                reference.syrk(alpha, a, transpose, beta, expected)
                blas.syrk(alpha, a, transpose, beta, actual)
                compareTriangle(expected, actual, n, lower = true, "syrk n=$n t=$transpose a=$alpha b=$beta")
            }
        }
    }
}

/** The unselected triangle starts as NaN, so a backend that writes outside the promised one fails. */
internal fun assertSyrkTriangleModesLeaveTheOtherTriangle(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260934)
    for (size in sizes) checkSyrkTriangle(blas, rng, rows = size, cols = maxOf(1, size - 2))
}

/** A symmetric rank-2k update agrees with the portable implementation and leaves the other triangle alone. */
internal fun assertSyr2kAgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20261031)
    for (size in sizes) {
        val rows = size
        val cols = maxOf(1, size - 2)
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, -0.5)) {
                        val a = randomMatrix(rows, cols, rng)
                        val b = randomMatrix(rows, cols, rng)
                        val n = if (transpose) cols else rows
                        val c0 = DoubleArray(n * n) { idx ->
                            if (!selects(lower, idx % n, idx / n) || beta == 0.0) {
                                Double.NaN
                            } else {
                                rng.nextDouble(-1.0, 1.0)
                            }
                        }
                        val expected = F64DenseMatrix.wrap(n, n, c0.copyOf())
                        val actual = F64DenseMatrix.wrap(n, n, c0.copyOf())
                        reference.syr2k(alpha, a, b, transpose, beta, expected, lower)
                        blas.syr2k(alpha, a, b, transpose, beta, actual, lower)
                        compareTriangle(
                            expected,
                            actual,
                            n,
                            lower,
                            "syr2k n=$n lower=$lower t=$transpose a=$alpha b=$beta",
                        )
                    }
                }
            }
        }
    }
}

private fun checkSyrkTriangle(blas: F64Blas, rng: Random, rows: Int, cols: Int) {
    for (lower in booleanArrayOf(true, false)) {
        for (transpose in booleanArrayOf(false, true)) {
            for (beta in doubleArrayOf(0.0, -0.5)) {
                val a = randomMatrix(rows, cols, rng)
                val n = if (transpose) cols else rows
                val c0 = DoubleArray(n * n) { idx ->
                    if (!selects(lower, idx % n, idx / n) || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                }
                val expected = F64DenseMatrix.wrap(n, n, c0.copyOf())
                val actual = F64DenseMatrix.wrap(n, n, c0.copyOf())
                reference.syrk(0.75, a, transpose, beta, expected, lower)
                blas.syrk(0.75, a, transpose, beta, actual, lower)
                compareTriangle(expected, actual, n, lower, "syrk lower=$lower t=$transpose b=$beta")
            }
        }
    }
}

private fun selects(lower: Boolean, i: Int, j: Int): Boolean = if (lower) j <= i else j >= i

/** The bound stays relative and tight, only widened by [n] for the accumulation the larger sizes do. */
private fun compareTriangle(expected: F64DenseMatrix, actual: F64DenseMatrix, n: Int, lower: Boolean, context: String) {
    for (i in 0 until n) {
        for (j in 0 until n) {
            val ctx = "$context ($i;$j)"
            if (selects(lower, i, j)) {
                val e = expected.data[i + j * n]
                val v = actual.data[i + j * n]
                assertTrue(abs(e - v) <= 1e-12 * n * maxOf(1.0, abs(e)), "$ctx: $e vs $v")
            } else {
                assertTrue(actual.data[i + j * n].isNaN(), "$ctx: untouched triangle written")
            }
        }
    }
}

/** The unselected triangle is NaN, so a backend that reads the mirror entry fails instead of agreeing. */
internal fun assertSymmetricProductsAgreeWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260912)
    for (lower in booleanArrayOf(true, false)) {
        for (n in sizes) {
            val (_, a) = poisonedSymmetric(rng, n, lower)
            val x = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val yExpected = DoubleArray(n) { Double.NaN }
            val yActual = DoubleArray(n) { Double.NaN }
            reference.symv(0.75, a, x, 0.0, yExpected, lower)
            blas.symv(0.75, a, x, 0.0, yActual, lower)
            assertClose(yExpected, yActual, "symv n=$n lower=$lower")
            checkSymm(blas, rng, a, n, lower)
        }
    }
}

private fun checkSymm(blas: F64Blas, rng: Random, a: F64DenseMatrix, n: Int, lower: Boolean) {
    val p = 3
    val b = randomMatrix(n, p, rng)
    val expected = F64DenseMatrix(n, p)
    val actual = F64DenseMatrix(n, p)
    reference.symm(0.75, a, b, 0.0, expected, lower)
    blas.symm(0.75, a, b, 0.0, actual, lower)
    assertClose(expected.data, actual.data, "symm n=$n lower=$lower")
    val br = randomMatrix(p, n, rng)
    val expectedRight = F64DenseMatrix(p, n)
    val actualRight = F64DenseMatrix(p, n)
    reference.symm(0.75, a, br, 0.0, expectedRight, lower, right = true)
    blas.symm(0.75, a, br, 0.0, actualRight, lower, right = true)
    assertClose(expectedRight.data, actualRight.data, "symm right n=$n lower=$lower")
}

/** The comparison is relative, since a determinant grows with the size of the matrix. */
internal fun assertDegenerateShapesFollowBlasQuickReturns(blas: F64Blas) {
    val y = DoubleArray(3) { Double.NaN }
    blas.gemv(1.0, F64DenseMatrix(0, 3), DoubleArray(0), 0.0, y, transpose = true)
    assertTrue(y.all(Double::isNaN), "gemv with a zero dimension changed ${y.toList()}")
    val stridedY = DoubleArray(3) { Double.NaN }
    blas.gemv(
        1.0,
        F64StridedMatrixView(0, 3, DoubleArray(0)),
        F64StridedVectorView(DoubleArray(0), 0, 0),
        0.0,
        F64StridedVectorView(stridedY, 2, 3, -1),
        transpose = true,
    )
    assertTrue(stridedY.all(Double::isNaN), "strided gemv with a zero dimension changed ${stridedY.toList()}")
    val c = F64DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
    blas.gemm(1.0, F64DenseMatrix(2, 0), false, F64DenseMatrix(0, 2), false, 0.0, c)
    assertTrue(c.data.all { it == 0.0 }, "gemm k=0 beta=0 left ${c.data.toList()}")
    val s = F64DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
    blas.syrk(0.0, F64DenseMatrix(2, 5), transpose = false, beta = 2.0, c = s)
    assertClose(doubleArrayOf(2.0, 4.0, 3.0, 8.0), s.data, "syrk alpha=0")
}

/** A 0x0 system has nothing to solve, so it yields an empty solution rather than failing. */
internal fun assertStridedProductsAgreeWithReference(blas: F64Blas) {
    val aData = DoubleArray(30) { -100.0 - it }
    val bData = DoubleArray(30) { -200.0 - it }
    val expectedData = DoubleArray(42) { -300.0 - it }
    val actualData = expectedData.copyOf()
    val a = F64StridedMatrixView(3, 2, aData, offset = 2, leadingDimension = 6)
    val b = F64StridedMatrixView(2, 4, bData, offset = 3, leadingDimension = 5)
    val expected = F64StridedMatrixView(3, 4, expectedData, offset = 7, leadingDimension = 7)
    val actual = F64StridedMatrixView(3, 4, actualData, offset = 7, leadingDimension = 7)
    for (j in 0 until a.cols) for (i in 0 until a.rows) a[i, j] = 1.0 + i + 2.0 * j
    for (j in 0 until b.cols) for (i in 0 until b.rows) b[i, j] = 0.5 + i - j

    reference.gemm(1.25, a, false, b, false, -0.5, expected)
    blas.gemm(1.25, a, false, b, false, -0.5, actual)

    assertClose(expectedData, actualData, "strided gemm", tolerance = 1e-12)

    val xData = doubleArrayOf(2.0, -99.0, 3.0, -99.0)
    val expectedY = doubleArrayOf(-1.0, -99.0, -2.0, -99.0, -3.0)
    val actualY = expectedY.copyOf()
    val x = F64StridedVectorView(xData, 0, 2, 2)
    val expectedVector = F64StridedVectorView(expectedY, 0, 3, 2)
    val actualVector = F64StridedVectorView(actualY, 0, 3, 2)

    reference.gemv(0.75, a, x, -0.25, expectedVector)
    blas.gemv(0.75, a, x, -0.25, actualVector)

    assertClose(expectedY, actualY, "strided gemv", tolerance = 1e-12)
}
