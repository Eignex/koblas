package com.eignex.koblas.sparse

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// What every host sparse-factorization binding owes the seam, written once so the JVM and native suites
// assert the same thing. Each function is one claim; the platform suites supply their own binding and keep
// whatever is theirs alone (loader, discovery, control-array settings).

/** A diagonally dominant system with roughly a quarter of the off-diagonal entries filled. */
internal fun sparseConformanceSystem(n: Int, rng: Random): SparseMatrix {
    val columns = ArrayList<List<Pair<Int, Double>>>(n)
    for (j in 0 until n) {
        val column = ArrayList<Pair<Int, Double>>()
        for (i in 0 until n) {
            val v = when {
                i == j -> n + 10.0
                rng.nextDouble() < 0.25 -> rng.nextDouble(-1.0, 1.0)
                else -> 0.0
            }
            if (v != 0.0) column.add(i to v)
        }
        columns.add(column)
    }
    return SparseMatrix.ofColumns(n, n, columns)
}

internal fun assertSolvesAgreeWithReference(lapack: SparseLapack) {
    val rng = Random(20260815)
    for (n in intArrayOf(1, 2, 7, 23, 60)) {
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val host = lapack.factor(a)
        val portable = ReferenceSparseLinearAlgebra.factor(a)
        assertTrue(!host.singular, "n=$n the host called a well-conditioned system singular")
        assertTrue(!portable.singular, "n=$n the reference called it singular")

        for (transpose in booleanArrayOf(false, true)) {
            val fromHost = host.solve(b, transpose)
            val fromPortable = portable.solve(b, transpose)
            assertClose(fromPortable, fromHost, "n=$n transpose=$transpose", tolerance = 1e-9)
            assertClose(b, a.gemv(fromHost, transpose), "n=$n transpose=$transpose residual", tolerance = 1e-9)
        }
    }
}

internal fun assertAliasedDestinationSolves(lapack: SparseLapack) {
    val rng = Random(20260816)
    val n = 12
    val a = sparseConformanceSystem(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    val f = lapack.factor(a)
    val expected = f.solve(b)
    val aliased = b.copyOf()
    f.solveInto(aliased, aliased)
    assertClose(expected, aliased, "aliased destination", tolerance = 1e-12)
}

internal fun assertDeterminantAgreesWithReference(lapack: SparseLapack) {
    val rng = Random(20260817)
    for (n in intArrayOf(1, 3, 8)) {
        val a = sparseConformanceSystem(n, rng)
        val host = lapack.factor(a).determinant()
        val portable = ReferenceSparseLinearAlgebra.factor(a).determinant()
        // The comparison is relative because the values grow with n.
        assertTrue(abs(host / portable - 1.0) < 1e-9, "n=$n determinant disagreed: $host vs $portable")
    }
}

/** A host that cannot name the failing pivot must say so rather than invent a position. */
internal fun assertSingularIsReportedWithUnknownPosition(lapack: SparseLapack) {
    val rank1 = SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 4.0)),
    )
    val f = lapack.factor(rank1)
    assertTrue(f.singular, "a rank-1 matrix should have been called singular")
    assertEquals(SINGULAR_POSITION_UNKNOWN, f.failedAt, "a host that cannot name the pivot must say so")
    assertEquals(0, f.nnz, "a singular factorization has no fill")
    assertEquals(0.0, f.determinant(), "a singular factorization has determinant zero")
    assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 1.0)) }
}

/** Equilibration and a drop tolerance are koblas's own, so a host must hand those requests back. */
internal fun assertUnsupportedRequestsFallBack(
    lapack: SparseLapack,
    hostFactorization: (SparseFactorization) -> Boolean,
) {
    val rng = Random(20260818)
    val a = sparseConformanceSystem(6, rng)
    val b = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
    for (fallback in listOf(lapack.factor(a, equilibrate = true), lapack.factor(a, dropTolerance = 1e-12))) {
        assertTrue(!hostFactorization(fallback), "a request the host cannot serve must not be ignored")
        assertClose(b, a.gemv(fallback.solve(b)), "fallback residual", tolerance = 1e-9)
    }
}

/** Native handles are freed per factorization, so a long loop must not grow without bound. */
internal fun assertRepeatedFactorizationsSurvive(lapack: SparseLapack) {
    val rng = Random(20260820)
    val a = sparseConformanceSystem(120, rng)
    var checksum = 0.0
    repeat(300) {
        val f = lapack.factor(a)
        checksum += abs(f.solve(DoubleArray(120) { 1.0 })[0])
    }
    assertTrue(checksum > 0.0, "the loop should have produced solutions")
}
