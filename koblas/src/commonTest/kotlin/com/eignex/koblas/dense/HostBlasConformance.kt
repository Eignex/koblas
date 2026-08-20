package com.eignex.koblas.dense

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.norm1
import com.eignex.koblas.poisonedIndefinite
import com.eignex.koblas.poisonedSymmetric
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.wellConditioned
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

private val reference = F64ReferenceLinearAlgebra

/** An SPD matrix with its strict upper triangle poisoned, since only the lower one may be read. */
internal fun poisonedSpd(n: Int, rng: Random): F64DenseMatrix {
    val seed = randomMatrix(n, n, rng)
    val a = F64DenseMatrix(n, n)
    reference.syrk(1.0, seed, transpose = false, beta = 0.0, c = a)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    for (i in 0 until n) for (j in i + 1 until n) a[i, j] = Double.NaN
    return a
}

/** Column [j] of R, padded to `m` entries, so that applying Q to it rebuilds column [j] of the input. */
internal fun rColumn(qr: F64QrDecomposition, j: Int): DoubleArray {
    val col = DoubleArray(qr.m)
    for (i in 0 until minOf(j + 1, qr.tau.size)) col[i] = qr.qr[i + j * qr.m]
    return col
}

internal fun assertLevel3AgreesWithReference(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260729)
    for (n in sizes) {
        val a = randomMatrix(n, n, rng)
        val b = randomMatrix(n, n, rng)
        assertClose(reference.gemm(a, b), blas.gemm(a, b), "gemm n=$n", tolerance = 1e-9 * n)
        for (uplo in listOf(Uplo.FULL, Uplo.LOWER)) {
            val expected = F64DenseMatrix(n, n)
            val actual = F64DenseMatrix(n, n)
            reference.syrk(1.0, a, transpose = true, beta = 0.0, c = expected, uplo = uplo)
            blas.syrk(1.0, a, transpose = true, beta = 0.0, c = actual, uplo = uplo)
            assertClose(expected, actual, "syrk $uplo n=$n", tolerance = 1e-9 * n)
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
            "trsm" to { la, m -> la.trsm(t, m, lower, transpose, unitDiag, right) },
            "trmm" to { la, m -> la.trmm(t, m, lower, transpose, unitDiag, right) },
        )) {
            val expected = F64DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(reference, it) }
            val actual = F64DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(blas, it) }
            assertClose(expected, actual, "$name right=$right $flags", tolerance = 1e-9)
        }
    }
}

/** The vector solve both ways, then the blocked multi-RHS path where a native trsm earns its call. */
internal fun assertLuAgreesWithReference(lapack: F64Lapack, sizes: IntArray) {
    val rng = Random(20260730)
    for (n in sizes) {
        val a = wellConditioned(n, rng) // diagonally dominant, so the solve is stable
        val rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val hostLu = lapack.factor(a)
        val portableLu = reference.factor(a)
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(
                reference.solve(portableLu, rhs, transpose),
                lapack.solve(hostLu, rhs, transpose),
                "LU solve n=$n t=$transpose",
                tolerance = 1e-9,
            )
        }
        val nrhs = 8
        val b = randomMatrix(n, nrhs, rng)
        assertClose(
            reference.solve(portableLu, b),
            lapack.solve(hostLu, b),
            "LU block solve n=$n",
            tolerance = 1e-9,
        )
    }
}

internal fun assertSpdSuiteAgreesWithReference(lapack: F64Lapack, sizes: IntArray) {
    val rng = Random(20260803)
    for (n in sizes) {
        val a = poisonedSpd(n, rng)

        val expected = reference.cholesky(a)
        val actual = lapack.cholesky(a)
        assertClose(expected.l, actual.l, "cholesky n=$n", tolerance = 1e-9)
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                assertEquals(0.0, actual.l[i, j], "cholesky n=$n left ${actual.l[i, j]} above the diagonal")
            }
        }

        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(reference.solve(expected, b), lapack.solve(actual, b), "solveSpd n=$n", tolerance = 1e-9)

        val inverse = lapack.invert(actual)
        assertClose(reference.invert(expected), inverse, "invertSpd n=$n", tolerance = 1e-7)
        for (i in 0 until n) {
            for (j in 0 until n) {
                assertEquals(inverse[i, j], inverse[j, i], 1e-12, "invertSpd n=$n is not symmetric at ($i,$j)")
            }
        }
    }
}

/**
 * A non-positive-definite input has no LAPACK equivalent, so the host hands the factorization back.
 * Keep [n] above the gate, since below it the host path is not involved.
 */
internal fun assertNonPositiveDefiniteFallsBack(lapack: F64Lapack, n: Int) {
    val bad = poisonedSpd(n, Random(20260808))
    bad[n - 1, n - 1] = -1.0 // breaks positive definiteness at the last leading minor
    val policy = CholeskyPolicy.Regularize()
    assertEquals(
        reference.cholesky(bad, policy).l[n - 1, n - 1],
        lapack.cholesky(bad, policy).l[n - 1, n - 1],
        1e-15,
    )
    assertFailsWith<IllegalArgumentException> { lapack.cholesky(bad) }
}

/**
 * `dsymv` derives its extent from one dimension and a leading dimension, so a non-square matrix has it read
 * `n²` entries from a shorter array, past the end of a pinned buffer that carries no bounds check. The shape
 * must be rejected whichever side of the level-2 gate the call lands on.
 */
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
internal fun assertSingularLdlIsRefused(lapack: F64Lapack, nrhs: Int = 4) {
    val ldl = lapack.ldl(F64DenseMatrix.zero(3, 3))
    assertTrue(ldl.singular, "a zero matrix factors to a singular LDL")
    assertFailsWith<SingularMatrix>("vector solve") { lapack.solve(ldl, DoubleArray(3)) }
    assertFailsWith<SingularMatrix>("$nrhs right-hand sides") { lapack.solve(ldl, F64DenseMatrix.zero(3, nrhs)) }
    assertFailsWith<SingularMatrix>("$nrhs right-hand sides into a destination") {
        lapack.solveInto(ldl, F64DenseMatrix.zero(3, nrhs), F64DenseMatrix.zero(3, nrhs))
    }
}

/**
 * `factorInto` promises the destination's own buffers, so a backend must not route it through a fresh
 * factorization and copy: that allocates the `n²` the caller passed a destination to avoid. Checked by
 * identity, since agreement with [F64Lapack.factor] alone would pass either way.
 */
internal fun assertFactorIntoUsesItsDestination(lapack: F64Lapack, n: Int) {
    val rng = Random(20260823)
    val first = wellConditioned(n, rng)
    val second = wellConditioned(n, rng)

    val out = lapack.factor(first)
    val luBuffer = out.lu
    val pivBuffer = out.piv
    val returned = lapack.factorInto(second, out)
    assertSame(out, returned, "factorInto must return its destination")
    assertSame(luBuffer, returned.lu, "factorInto must keep the destination's factor buffer")
    assertSame(pivBuffer, returned.piv, "factorInto must keep the destination's pivot buffer")

    val fresh = lapack.factor(second)
    assertClose(fresh.lu, returned.lu, "factorInto factors n=$n", tolerance = 1e-12)
    assertEquals(fresh.piv.toList(), returned.piv.toList(), "factorInto pivots n=$n")
    assertEquals(fresh.failedAt, returned.failedAt, "factorInto singularity n=$n")

    // A singular matrix must update failedAt, not keep the previous factorization's verdict.
    val singular = F64DenseMatrix(n, n)
    lapack.factorInto(singular, returned)
    assertTrue(returned.singular, "factorInto must report a singular refactorization")
    lapack.factorInto(second, returned)
    assertEquals(fresh.failedAt, returned.failedAt, "factorInto must clear a stale singular verdict")
}

/** The destination starts as NaN wherever beta is zero, so a backend that reads it instead of writing it fails. */
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

/** A rank update of a symmetric matrix is symmetric to the last bit, not just to a tolerance. */
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
                val c0 = DoubleArray(n * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                val expected = F64DenseMatrix.wrap(n, n, c0.copyOf())
                val actual = F64DenseMatrix.wrap(n, n, c0.copyOf())
                reference.syrk(alpha, a, transpose, beta, expected)
                blas.syrk(alpha, a, transpose, beta, actual)
                assertClose(expected.data, actual.data, "syrk n=$n t=$transpose a=$alpha b=$beta")
                if (beta == 0.0) assertExactlySymmetric(actual, n, "syrk n=$n t=$transpose")
            }
        }
    }
}

private fun assertExactlySymmetric(c: F64DenseMatrix, n: Int, context: String) {
    for (i in 0 until n) {
        for (j in 0 until i) {
            assertEquals(c.data[i + j * n], c.data[j + i * n], "$context asymmetric at ($i;$j)")
        }
    }
}

/** The unselected triangle starts as NaN, so a backend that writes outside the promised one fails. */
internal fun assertSyrkTriangleModesLeaveTheOtherTriangle(blas: F64Blas, sizes: IntArray) {
    val rng = Random(20260934)
    for (size in sizes) checkSyrkTriangle(blas, rng, rows = size, cols = maxOf(1, size - 2))
}

private fun checkSyrkTriangle(blas: F64Blas, rng: Random, rows: Int, cols: Int) {
    for (uplo in listOf(Uplo.LOWER, Uplo.UPPER)) {
        for (transpose in booleanArrayOf(false, true)) {
            for (beta in doubleArrayOf(0.0, -0.5)) {
                val a = randomMatrix(rows, cols, rng)
                val n = if (transpose) cols else rows
                val c0 = DoubleArray(n * n) { idx ->
                    if (!selects(uplo, idx % n, idx / n) || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                }
                val expected = F64DenseMatrix.wrap(n, n, c0.copyOf())
                val actual = F64DenseMatrix.wrap(n, n, c0.copyOf())
                reference.syrk(0.75, a, transpose, beta, expected, uplo)
                blas.syrk(0.75, a, transpose, beta, actual, uplo)
                compareTriangle(expected, actual, n, uplo, "syrk $uplo t=$transpose b=$beta")
            }
        }
    }
}

private fun selects(uplo: Uplo, i: Int, j: Int): Boolean = if (uplo == Uplo.LOWER) j <= i else j >= i

/** The bound stays relative and tight, only widened by [n] for the accumulation the larger sizes do. */
private fun compareTriangle(expected: F64DenseMatrix, actual: F64DenseMatrix, n: Int, uplo: Uplo, context: String) {
    for (i in 0 until n) {
        for (j in 0 until n) {
            val ctx = "$context ($i;$j)"
            if (selects(uplo, i, j)) {
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
internal fun assertDeterminantAgreesWithReference(lapack: F64Lapack, sizes: IntArray) {
    val rng = Random(20260730)
    for (n in sizes) {
        val a = wellConditioned(n, rng)
        val expected = reference.factor(a).determinant()
        val actual = lapack.factor(a).determinant()
        assertTrue(
            abs(expected - actual) <= 1e-9 * maxOf(1.0, abs(expected)),
            "determinant n=$n: $expected vs $actual",
        )
    }
}

internal fun assertSingularLuIsFlagged(lapack: F64Lapack) {
    val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    val lu = lapack.factor(a)
    assertTrue(lu.singular, "a rank-1 matrix factors to a singular LU")
    assertEquals(0.0, lu.determinant(), "a singular factorization has determinant zero")
}

/** A factorization is a value, so either backend must be able to solve with the other one's factors. */
internal fun assertLuFactorsInterchange(lapack: F64Lapack, n: Int) {
    val rng = Random(20260731)
    val a = wellConditioned(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    for (transpose in booleanArrayOf(false, true)) {
        val viaHostFactor = reference.solve(lapack.factor(a), b, transpose)
        val viaReferenceFactor = lapack.solve(reference.factor(a), b, transpose)
        assertClose(viaHostFactor, viaReferenceFactor, "interchange n=$n t=$transpose", tolerance = 1e-10)
    }
}

/** An estimator is free to disagree in detail, so only the order of magnitude is compared. */
internal fun assertRcondAgreesWithReference(lapack: F64Lapack, sizes: IntArray) {
    val rng = Random(20260905)
    for (n in sizes) {
        val a = wellConditioned(n, rng)
        val anorm = norm1(a)
        val expected = reference.rcond(reference.factor(a), anorm)
        val actual = lapack.rcond(lapack.factor(a), anorm)
        assertTrue(actual in expected / 10.0..expected * 10.0, "n=$n: host $actual vs reference $expected")
    }
    val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    assertEquals(0.0, lapack.rcond(lapack.factor(singular), norm1(singular)))
    assertEquals(1.0, lapack.rcond(lapack.factor(F64DenseMatrix(0, 0)), 0.0))
}

/**
 * The block solve of an indefinite system, with the unselected triangle poisoned. [nrhs] must reach the width
 * at which the native multi-right-hand-side path takes over.
 */
internal fun assertLdlBlockSolveAgreesWithReference(lapack: F64Lapack, n: Int, nrhs: Int) {
    val rng = Random(20260952)
    val b = randomMatrix(n, nrhs, rng)
    val (_, a) = poisonedIndefinite(rng, n)
    val expected = reference.solve(reference.ldl(a), b)
    val actual = lapack.solve(lapack.ldl(a), b)
    assertClose(expected.data, actual.data, "ldl block n=$n nrhs=$nrhs", tolerance = 1e-9)
}

internal fun assertLdlFactorsInterchange(lapack: F64Lapack, sizes: IntArray) {
    val rng = Random(20260932)
    for (n in sizes) {
        val (_, a) = poisonedIndefinite(rng, n)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val fReference = reference.ldl(a)
        val fHost = lapack.ldl(a)
        val xs = listOf(
            reference.solve(fReference, b),
            reference.solve(fHost, b),
            lapack.solve(fReference, b),
            lapack.solve(fHost, b),
        )
        for ((i, x) in xs.withIndex()) {
            assertClose(xs[0], x, "ldl interchange n=$n variant $i", tolerance = 1e-9)
        }
    }
    assertTrue(lapack.ldl(F64DenseMatrix(3, 3)).singular, "a zero matrix factors to a singular LDL")
    assertTrue(lapack.solve(lapack.ldl(F64DenseMatrix(0, 0)), DoubleArray(0)).isEmpty(), "an empty solve is empty")
}

/** Both least squares and the minimum-norm solve, across every pairing of the two backends' factors. */
internal fun assertQrFactorsInterchange(lapack: F64Lapack, shapes: List<Pair<Int, Int>>) {
    // A column whose tail is already zero is where the two can differ without either being wrong: both
    // factor it, and only a zero reflector leaves R's diagonal as it found it. Sized past the LAPACK
    // dispatch threshold so the binding rather than the fallback answers.
    val triangular = F64DenseMatrix.diagonal(80)
    assertClose(reference.qr(triangular).tau, lapack.qr(triangular).tau, "tau on a triangular operand")
    assertClose(reference.qr(triangular).qr, lapack.qr(triangular).qr, "R on a triangular operand")
    val rng = Random(20260925)
    for ((m, n) in shapes) {
        val a = randomMatrix(m, n, rng)
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val fReference = reference.qr(a)
        val fHost = lapack.qr(a)
        val xs = listOf(
            reference.solveLeastSquares(fReference, b),
            reference.solveLeastSquares(fHost, b),
            lapack.solveLeastSquares(fReference, b),
            lapack.solveLeastSquares(fHost, b),
        )
        // The operand is a random matrix rather than a dominant one, so its conditioning worsens with the
        // shape and a flat bound would be tight only at the largest one. Scaled so the smallest shapes keep
        // the bound they had.
        val leastSquaresTolerance = 1e-10 * maxOf(1.0, m / 16.0)
        for ((i, x) in xs.withIndex()) {
            assertClose(xs[0], x, "qr interchange ${m}x$n variant $i", tolerance = leastSquaresTolerance)
        }
        val y = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            reference.applyQ(fHost, y),
            lapack.applyQ(fHost, y),
            "applyQ on host factors ${m}x$n",
            tolerance = 1e-11 * maxOf(1.0, m / 16.0),
        )
        assertClose(
            reference.applyQ(fReference, y),
            lapack.applyQ(fReference, y),
            "applyQ on reference factors ${m}x$n",
            tolerance = 1e-11,
        )
        val bWide = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            reference.solveMinimumNorm(fReference, bWide),
            lapack.solveMinimumNorm(fHost, bWide),
            "solveMinimumNorm ${n}x$m",
            tolerance = 1e-10,
        )
    }
}

/** With beta zero the destination is written rather than read, so an empty sum has to leave exact zeros. */
internal fun assertDegenerateShapesHonorTheBetaConventions(blas: F64Blas) {
    val y = DoubleArray(3) { Double.NaN }
    blas.gemv(1.0, F64DenseMatrix(0, 3), DoubleArray(0), 0.0, y, transpose = true)
    assertTrue(y.all { it == 0.0 }, "gemv beta=0 with an empty sum left ${y.toList()}")
    val c = F64DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
    blas.gemm(1.0, F64DenseMatrix(2, 0), false, F64DenseMatrix(0, 2), false, 0.0, c)
    assertTrue(c.data.all { it == 0.0 }, "gemm k=0 beta=0 left ${c.data.toList()}")
    val s = F64DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
    blas.syrk(0.0, F64DenseMatrix(2, 5), transpose = false, beta = 2.0, c = s)
    assertClose(doubleArrayOf(2.0, 4.0, 6.0, 8.0), s.data, "syrk alpha=0")
}

/** A 0x0 system has nothing to solve, so it yields an empty solution rather than failing. */
internal fun assertAnEmptyFactorizationSolvesEmpty(lapack: F64Lapack) {
    val empty = lapack.factor(F64DenseMatrix(0, 0))
    assertEquals(0, lapack.solve(empty, DoubleArray(0)).size)
}

/**
 * Pivoted QR against the reference: the rank of a deliberately deficient matrix, that the pivots are a
 * permutation, that `A·P = Q·R` column by column, and that the pivoted least-squares solve agrees.
 *
 * Each rank in [ranks] builds an `m x cols` matrix as a product of full-rank factors of that rank, so the
 * deficiency is exact rather than the result of rounding.
 */
internal fun assertPivotedQrAgreesWithReference(lapack: F64Lapack, m: Int, cols: Int, ranks: IntArray) {
    val rng = Random(20260809)
    for (rank in ranks) {
        val left = randomMatrix(m, rank, rng)
        val right = randomMatrix(rank, cols, rng)
        val a = F64DenseMatrix(m, cols)
        reference.gemm(1.0, left, false, right, false, 0.0, a)

        val expected = reference.qrPivoted(a)
        val actual = lapack.qrPivoted(a)
        assertEquals(expected.rank, actual.rank, "rank of a ${m}x$cols matrix built at rank $rank")
        assertEquals((0 until cols).toList(), actual.pivots.sorted(), "pivots must be a permutation")

        for (j in 0 until cols) {
            val rebuilt = lapack.applyQ(actual.factorization, rColumn(actual.factorization, j))
            val original = DoubleArray(m) { i -> a[i, actual.pivots[j]] }
            assertClose(original, rebuilt, tolerance = 1e-8, context = "A·P = Q·R rank=$rank column $j")
        }

        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            reference.solveLeastSquares(expected, b),
            lapack.solveLeastSquares(actual, b),
            tolerance = 1e-7,
            context = "pivoted least squares rank=$rank",
        )
    }
}
