package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

private val reference = ReferenceLinearAlgebra

/** An SPD matrix with its strict upper triangle poisoned, since only the lower one may be read. */
internal fun poisonedSpd(n: Int, rng: Random): DenseMatrix {
    val seed = randomMatrix(n, n, rng)
    val a = DenseMatrix(n, n)
    reference.syrk(1.0, seed, transpose = false, beta = 0.0, c = a)
    for (i in 0 until n) a[i, i] = a[i, i] + n
    for (i in 0 until n) for (j in i + 1 until n) a[i, j] = Double.NaN
    return a
}

internal fun assertLevel3AgreesWithReference(blas: Blas, sizes: IntArray) {
    val rng = Random(20260729)
    for (n in sizes) {
        val a = randomMatrix(n, n, rng)
        val b = randomMatrix(n, n, rng)
        assertClose(reference.gemm(a, b), blas.gemm(a, b), "gemm n=$n", tolerance = 1e-9 * n)
        for (uplo in listOf(Uplo.FULL, Uplo.LOWER)) {
            val expected = DenseMatrix(n, n)
            val actual = DenseMatrix(n, n)
            reference.syrk(1.0, a, transpose = true, beta = 0.0, c = expected, uplo = uplo)
            blas.syrk(1.0, a, transpose = true, beta = 0.0, c = actual, uplo = uplo)
            assertClose(expected, actual, "syrk $uplo n=$n", tolerance = 1e-9 * n)
        }
    }
}

internal fun assertGerAgreesWithReference(blas: Blas) {
    val rng = Random(20260802)
    for ((rows, cols) in listOf(1 to 1, 4 to 7, 9 to 3, 30 to 9)) {
        for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
            val x = DoubleArray(rows) { rng.nextDouble(-1.0, 1.0) }
            val y = DoubleArray(cols) { rng.nextDouble(-1.0, 1.0) }
            val a0 = randomMatrix(rows, cols, rng)
            val expected = DenseMatrix(rows, cols, a0.data.copyOf())
            reference.ger(alpha, x, y, expected)
            val actual = DenseMatrix(rows, cols, a0.data.copyOf())
            blas.ger(alpha, x, y, actual)
            assertClose(expected, actual, "ger ${rows}x$cols alpha=$alpha")
        }
    }
}

/** The unselected triangle is NaN, so reading outside the promised one fails the comparison. */
internal fun assertTriangularAgreesWithReference(blas: Blas, sizes: IntArray) {
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
private fun checkTriangular(blas: Blas, rng: Random, n: Int, lower: Boolean, transpose: Boolean, unitDiag: Boolean) {
    val t = DenseMatrix(n, n)
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
    for ((name, op) in listOf<Pair<String, (Blas, DoubleArray) -> Unit>>(
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
        for ((name, op) in listOf<Pair<String, (Blas, DenseMatrix) -> Unit>>(
            "trsm" to { la, m -> la.trsm(t, m, lower, transpose, unitDiag, right) },
            "trmm" to { la, m -> la.trmm(t, m, lower, transpose, unitDiag, right) },
        )) {
            val expected = DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(reference, it) }
            val actual = DenseMatrix(b.rows, b.cols, b.data.copyOf()).also { op(blas, it) }
            assertClose(expected, actual, "$name right=$right $flags", tolerance = 1e-9)
        }
    }
}

/** The vector solve both ways, then the blocked multi-RHS path where a native trsm earns its call. */
internal fun assertLuAgreesWithReference(lapack: Lapack, sizes: IntArray) {
    val rng = Random(20260730)
    for (n in sizes) {
        val a = randomMatrix(n, n, rng)
        for (i in 0 until n) a[i, i] = a[i, i] + n // diagonally dominant, so the solve is stable
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

internal fun assertSpdSuiteAgreesWithReference(lapack: Lapack, sizes: IntArray) {
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
internal fun assertNonPositiveDefiniteFallsBack(lapack: Lapack, n: Int) {
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
internal fun assertSymvRefusesNonSquare(blas: Blas) {
    for (n in intArrayOf(2, 17, 64, 129)) {
        assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
            blas.symv(1.0, DenseMatrix.zero(n, n + 1), DoubleArray(n), 0.0, DoubleArray(n))
        }
        assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
            blas.symv(1.0, DenseMatrix.zero(n + 1, n), DoubleArray(n + 1), 0.0, DoubleArray(n + 1))
        }
    }
}

/**
 * `dsytrs` divides by a zero pivot and still reports success, so a backend that skips the singularity check
 * returns infinities where it should throw. [nrhs] must reach the width at which the native
 * multi-right-hand-side path takes over.
 */
internal fun assertSingularLdlIsRefused(lapack: Lapack, nrhs: Int = 4) {
    val ldl = lapack.ldl(DenseMatrix.zero(3, 3))
    assertTrue(ldl.singular, "a zero matrix factors to a singular LDL")
    assertFailsWith<SingularMatrix>("vector solve") { lapack.solve(ldl, DoubleArray(3)) }
    assertFailsWith<SingularMatrix>("$nrhs right-hand sides") { lapack.solve(ldl, DenseMatrix.zero(3, nrhs)) }
    assertFailsWith<SingularMatrix>("$nrhs right-hand sides into a destination") {
        lapack.solveInto(ldl, DenseMatrix.zero(3, nrhs), DenseMatrix.zero(3, nrhs))
    }
}

/**
 * `factorInto` promises the destination's own buffers, so a backend must not route it through a fresh
 * factorization and copy: that allocates the `n²` the caller passed a destination to avoid. Checked by
 * identity, since agreement with [Lapack.factor] alone would pass either way.
 */
internal fun assertFactorIntoUsesItsDestination(lapack: Lapack, n: Int) {
    val rng = Random(20260823)
    val first = randomMatrix(n, n, rng)
    for (i in 0 until n) first[i, i] = first[i, i] + n
    val second = randomMatrix(n, n, rng)
    for (i in 0 until n) second[i, i] = second[i, i] + n

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
    val singular = DenseMatrix(n, n)
    lapack.factorInto(singular, returned)
    assertTrue(returned.singular, "factorInto must report a singular refactorization")
    lapack.factorInto(second, returned)
    assertEquals(fresh.failedAt, returned.failedAt, "factorInto must clear a stale singular verdict")
}
