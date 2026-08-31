package com.eignex.koblas.dense.host

import com.eignex.koblas.*
import com.eignex.koblas.core.*
import com.eignex.koblas.dense.*
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

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
internal fun assertLuAgreesWithReference(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260730)
    for (n in sizes) {
        val a = wellConditioned(n, rng) // diagonally dominant, so the solve is stable
        val rhs = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val hostLu = decompositions.factor(a)
        val portableLu = reference.factor(a)
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(
                reference.solve(portableLu, rhs, transpose),
                decompositions.solve(hostLu, rhs, transpose),
                "LU solve n=$n t=$transpose",
                tolerance = 1e-9,
            )
        }
        val nrhs = 8
        val b = randomMatrix(n, nrhs, rng)
        assertClose(
            reference.solve(portableLu, b),
            decompositions.solve(hostLu, b),
            "LU block solve n=$n",
            tolerance = 1e-9,
        )
    }
}

/**
 * The LU inverse and the triangular inverse. Both hold the reference's conventions as well as its numbers:
 * an inverse of a singular factor is refused, and a triangular inverse returns its triangle over zeros with
 * a unit diagonal written in.
 */
internal fun assertInversesAgreeWithReference(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260826)
    for (n in sizes) {
        val dominant = wellConditioned(n, rng)
        // Row-reversed as well as dominant: a diagonally dominant matrix factors to an identity pivot, so
        // the permutation an inverse has to undo is only exercised by a matrix that actually swaps rows.
        for (a in listOf(dominant, rowsReversed(dominant))) {
            assertClose(
                reference.invert(reference.factor(a)),
                decompositions.invert(decompositions.factor(a)),
                "invert lu n=$n",
                tolerance = 1e-8,
            )
        }
        val a = dominant
        for (lower in booleanArrayOf(true, false)) {
            for (unitDiag in booleanArrayOf(false, true)) {
                val t = triangleOf(a, lower, unitDiag)
                assertClose(
                    reference.trtri(t, lower, unitDiag),
                    decompositions.trtri(t, lower, unitDiag),
                    "trtri n=$n lower=$lower unit=$unitDiag",
                    tolerance = 1e-8,
                )
            }
        }
    }
    val singular = F64DenseMatrix(4, 4)
    assertFailsWith<SingularMatrix>("a singular factor has no inverse") {
        decompositions.invert(decompositions.factor(singular))
    }
}

/** [a] with its row order reversed, so a factorization of it has to pivot. */
private fun rowsReversed(a: F64DenseMatrix): F64DenseMatrix {
    val n = a.rows
    val out = F64DenseMatrix(n, n)
    for (j in 0 until n) for (i in 0 until n) out[i, j] = a[n - 1 - i, j]
    return out
}

/** The [lower] or upper triangle of [a] over zeros, with the diagonal replaced when [unitDiag]. */
private fun triangleOf(a: F64DenseMatrix, lower: Boolean, unitDiag: Boolean): F64DenseMatrix {
    val n = a.rows
    val t = F64DenseMatrix(n, n)
    for (j in 0 until n) {
        for (i in 0 until n) {
            if (if (lower) i >= j else i <= j) t[i, j] = a[i, j]
        }
        t[j, j] = if (unitDiag) 1.0 else a[j, j]
    }
    return t
}

internal fun assertSpdSuiteAgreesWithReference(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260803)
    for (n in sizes) {
        val a = poisonedSpd(n, rng)

        val expected = reference.cholesky(a)
        val actual = decompositions.cholesky(a)
        assertClose(expected.l, actual.l, "cholesky n=$n", tolerance = 1e-9)
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                assertEquals(0.0, actual.l[i, j], "cholesky n=$n left ${actual.l[i, j]} above the diagonal")
            }
        }

        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(reference.solve(expected, b), decompositions.solve(actual, b), "solveSpd n=$n", tolerance = 1e-9)

        val inverse = decompositions.invert(actual)
        assertClose(reference.invert(expected), inverse, "invertSpd n=$n", tolerance = 1e-7)
        for (i in 0 until n) {
            for (j in 0 until n) {
                assertEquals(inverse[i, j], inverse[j, i], 1e-12, "invertSpd n=$n is not symmetric at ($i,$j)")
            }
        }

        // The factor is the caller's to build over any square matrix, so a singular one reaches here and the
        // two halves have to answer it alike rather than one of them throwing.
        val singular = F64CholeskyDecomposition(F64DenseMatrix.zero(n, n))
        val referenceEntry = reference.invert(singular).data[0]
        assertEquals(
            referenceEntry.isFinite(),
            decompositions.invert(singular).data[0].isFinite(),
            "invertSpd n=$n disagrees with the reference on a singular factor",
        )
    }
}

/**
 * A non-positive-definite input has no LAPACK equivalent, so the host hands the factorization back.
 * Uses a moderate [n] so the host path performs representative work.
 */
internal fun assertNonPositiveDefiniteFallsBack(decompositions: F64Decompositions, n: Int) {
    val bad = poisonedSpd(n, Random(20260808))
    bad[n - 1, n - 1] = -1.0 // breaks positive definiteness at the last leading minor
    val policy = CholeskyPolicy.Regularize()
    assertEquals(
        reference.cholesky(bad, policy).l[n - 1, n - 1],
        decompositions.cholesky(bad, policy).l[n - 1, n - 1],
        1e-15,
    )
    assertFailsWith<IllegalArgumentException> { decompositions.cholesky(bad) }
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
internal fun assertSingularLdlIsRefused(decompositions: F64Decompositions, nrhs: Int = 4) {
    val ldl = decompositions.ldl(F64DenseMatrix.zero(3, 3))
    assertTrue(ldl.singular, "a zero matrix factors to a singular LDL")
    assertFailsWith<SingularMatrix>("vector solve") { decompositions.solve(ldl, DoubleArray(3)) }
    assertFailsWith<SingularMatrix>(
        "$nrhs right-hand sides",
    ) { decompositions.solve(ldl, F64DenseMatrix.zero(3, nrhs)) }
    assertFailsWith<SingularMatrix>("$nrhs right-hand sides into a destination") {
        decompositions.solveInto(ldl, F64DenseMatrix.zero(3, nrhs), F64DenseMatrix.zero(3, nrhs))
    }
}

/**
 * `factorInto` promises the destination's own buffers, so a backend must not route it through a fresh
 * factorization and copy: that allocates the `n²` the caller passed a destination to avoid. Checked by
 * identity, since agreement with [F64Decompositions.factor] alone would pass either way.
 */
internal fun assertFactorIntoUsesItsDestination(decompositions: F64Decompositions, n: Int) {
    val rng = Random(20260823)
    val first = wellConditioned(n, rng)
    val second = wellConditioned(n, rng)

    val out = decompositions.factor(first)
    val luBuffer = out.lu
    val pivBuffer = out.piv
    val returned = decompositions.factorInto(second, out)
    assertSame(out, returned, "factorInto must return its destination")
    assertSame(luBuffer, returned.lu, "factorInto must keep the destination's factor buffer")
    assertSame(pivBuffer, returned.piv, "factorInto must keep the destination's pivot buffer")

    val fresh = decompositions.factor(second)
    assertClose(fresh.lu, returned.lu, "factorInto factors n=$n", tolerance = 1e-12)
    assertEquals(fresh.piv.toList(), returned.piv.toList(), "factorInto pivots n=$n")
    assertEquals(fresh.failedAt, returned.failedAt, "factorInto singularity n=$n")

    // A singular matrix must update failedAt, not keep the previous factorization's verdict.
    val singular = F64DenseMatrix(n, n)
    decompositions.factorInto(singular, returned)
    assertTrue(returned.singular, "factorInto must report a singular refactorization")
    decompositions.factorInto(second, returned)
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
internal fun assertDeterminantAgreesWithReference(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260730)
    for (n in sizes) {
        val a = wellConditioned(n, rng)
        val expected = reference.factor(a).determinant()
        val actual = decompositions.factor(a).determinant()
        assertTrue(
            abs(expected - actual) <= 1e-9 * maxOf(1.0, abs(expected)),
            "determinant n=$n: $expected vs $actual",
        )
    }
}

internal fun assertSingularLuIsFlagged(decompositions: F64Decompositions) {
    val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    val lu = decompositions.factor(a)
    assertTrue(lu.singular, "a rank-1 matrix factors to a singular LU")
    assertEquals(0.0, lu.determinant(), "a singular factorization has determinant zero")
}

/** A factorization is a value, so either backend must be able to solve with the other one's factors. */
internal fun assertLuFactorsInterchange(decompositions: F64Decompositions, n: Int) {
    val rng = Random(20260731)
    val a = wellConditioned(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    for (transpose in booleanArrayOf(false, true)) {
        val viaHostFactor = reference.solve(decompositions.factor(a), b, transpose)
        val viaReferenceFactor = decompositions.solve(reference.factor(a), b, transpose)
        assertClose(viaHostFactor, viaReferenceFactor, "interchange n=$n t=$transpose", tolerance = 1e-10)
    }
}

/** An estimator is free to disagree in detail, so only the order of magnitude is compared. */
internal fun assertRcondAgreesWithReference(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260905)
    for (n in sizes) {
        val a = wellConditioned(n, rng)
        val anorm = a.norm1()
        val expected = reference.rcond(reference.factor(a), anorm)
        val actual = decompositions.rcond(decompositions.factor(a), anorm)
        assertTrue(actual in expected / 10.0..expected * 10.0, "n=$n: host $actual vs reference $expected")
    }
    val singular = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
    assertEquals(0.0, decompositions.rcond(decompositions.factor(singular), singular.norm1()))
    assertEquals(1.0, decompositions.rcond(decompositions.factor(F64DenseMatrix(0, 0)), 0.0))
}

/**
 * The block solve of an indefinite system, with the unselected triangle poisoned. [nrhs] must reach the width
 * at which the native multi-right-hand-side path takes over.
 */
internal fun assertLdlBlockSolveAgreesWithReference(decompositions: F64Decompositions, n: Int, nrhs: Int) {
    val rng = Random(20260952)
    val b = randomMatrix(n, nrhs, rng)
    val (_, a) = poisonedIndefinite(rng, n)
    val expected = reference.solve(reference.ldl(a), b)
    val actual = decompositions.solve(decompositions.ldl(a), b)
    assertClose(expected.data, actual.data, "ldl block n=$n nrhs=$nrhs", tolerance = 1e-9)
}

internal fun assertLdlFactorsInterchange(decompositions: F64Decompositions, sizes: IntArray) {
    val rng = Random(20260932)
    for (n in sizes) {
        val (_, a) = poisonedIndefinite(rng, n)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val fReference = reference.ldl(a)
        val fHost = decompositions.ldl(a)
        val xs = listOf(
            reference.solve(fReference, b),
            reference.solve(fHost, b),
            decompositions.solve(fReference, b),
            decompositions.solve(fHost, b),
        )
        for ((i, x) in xs.withIndex()) {
            assertClose(xs[0], x, "ldl interchange n=$n variant $i", tolerance = 1e-9)
        }
    }
    assertTrue(decompositions.ldl(F64DenseMatrix(3, 3)).singular, "a zero matrix factors to a singular LDL")
    assertTrue(
        decompositions.solve(decompositions.ldl(F64DenseMatrix(0, 0)), DoubleArray(0)).isEmpty(),
        "an empty solve is empty",
    )
}

/** Both least squares and the minimum-norm solve, across every pairing of the two backends' factors. */
internal fun assertQrFactorsInterchange(decompositions: F64Decompositions, shapes: List<Pair<Int, Int>>) {
    // A column whose tail is already zero is where the two can differ without either being wrong: both
    // factor it, and only a zero reflector leaves R's diagonal as it found it. Sized past the LAPACK
    // size used by every binding conformance run.
    val triangular = F64DenseMatrix.diagonal(80)
    assertClose(reference.qr(triangular).tau, decompositions.qr(triangular).tau, "tau on a triangular operand")
    assertClose(reference.qr(triangular).qr, decompositions.qr(triangular).qr, "R on a triangular operand")
    val rng = Random(20260925)
    for ((m, n) in shapes) {
        val a = randomMatrix(m, n, rng)
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val fReference = reference.qr(a)
        val fHost = decompositions.qr(a)
        val xs = listOf(
            reference.solve(fReference, b),
            reference.solve(fHost, b),
            decompositions.solve(fReference, b),
            decompositions.solve(fHost, b),
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
            decompositions.applyQ(fHost, y),
            "applyQ on host factors ${m}x$n",
            tolerance = 1e-11 * maxOf(1.0, m / 16.0),
        )
        assertClose(
            reference.applyQ(fReference, y),
            decompositions.applyQ(fReference, y),
            "applyQ on reference factors ${m}x$n",
            tolerance = 1e-11,
        )
        val bWide = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            reference.solve(fReference, bWide, minimumNorm = true),
            decompositions.solve(fHost, bWide, minimumNorm = true),
            "minimum-norm solve ${n}x$m",
            tolerance = 1e-10,
        )
    }
}

/** A zero matrix dimension takes the BLAS quick return before beta is applied. */
internal fun assertDegenerateShapesFollowBlasQuickReturns(blas: F64Blas) {
    val y = DoubleArray(3) { Double.NaN }
    blas.gemv(1.0, F64DenseMatrix(0, 3), DoubleArray(0), 0.0, y, transpose = true)
    assertTrue(y.all(Double::isNaN), "gemv with a zero dimension changed ${y.toList()}")
    val c = F64DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
    blas.gemm(1.0, F64DenseMatrix(2, 0), false, F64DenseMatrix(0, 2), false, 0.0, c)
    assertTrue(c.data.all { it == 0.0 }, "gemm k=0 beta=0 left ${c.data.toList()}")
    val s = F64DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, 2.0, 3.0, 4.0))
    blas.syrk(0.0, F64DenseMatrix(2, 5), transpose = false, beta = 2.0, c = s)
    assertClose(doubleArrayOf(2.0, 4.0, 3.0, 8.0), s.data, "syrk alpha=0")
}

/** A 0x0 system has nothing to solve, so it yields an empty solution rather than failing. */
internal fun assertAnEmptyFactorizationSolvesEmpty(decompositions: F64Decompositions) {
    val empty = decompositions.factor(F64DenseMatrix(0, 0))
    assertEquals(0, decompositions.solve(empty, DoubleArray(0)).size)
}

/** Offset, stride, and leading-dimension calls through the same panels as the portable implementation. */
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

/**
 * Pivoted QR against the reference: the rank of a deliberately deficient matrix, that the pivots are a
 * permutation, that `A·P = Q·R` column by column, and that the pivoted least-squares solve agrees.
 *
 * Each rank in [ranks] builds an `m x cols` matrix as a product of full-rank factors of that rank, so the
 * deficiency is exact rather than the result of rounding.
 */
internal fun assertPivotedQrAgreesWithReference(decompositions: F64Decompositions, m: Int, cols: Int, ranks: IntArray) {
    val rng = Random(20260809)
    for (rank in ranks) {
        val left = randomMatrix(m, rank, rng)
        val right = randomMatrix(rank, cols, rng)
        val a = F64DenseMatrix(m, cols)
        reference.gemm(1.0, left, false, right, false, 0.0, a)

        val expected = reference.qrPivoted(a)
        val actual = decompositions.qrPivoted(a)
        assertEquals(expected.rank, actual.rank, "rank of a ${m}x$cols matrix built at rank $rank")
        assertEquals((0 until cols).toList(), actual.pivots.sorted(), "pivots must be a permutation")

        for (j in 0 until cols) {
            val rebuilt = decompositions.applyQ(actual.factorization, rColumn(actual.factorization, j))
            val original = DoubleArray(m) { i -> a[i, actual.pivots[j]] }
            assertClose(original, rebuilt, tolerance = 1e-8, context = "A·P = Q·R rank=$rank column $j")
        }

        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            reference.solve(expected, b),
            decompositions.solve(actual, b),
            tolerance = 1e-7,
            context = "pivoted least squares rank=$rank",
        )
    }
}
