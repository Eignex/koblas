package com.eignex.koblas.sparse

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import com.eignex.koblas.withCleanBackends
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** A diagonally dominant system with roughly a quarter of the off-diagonal entries filled. */
internal fun sparseConformanceSystem(n: Int, rng: Random): F64SparseMatrix {
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
    return F64SparseMatrix.ofColumns(n, n, columns)
}

/** A·x computed straight from the CSC arrays, so no seam is involved in checking a seam. */
internal fun multiply(a: F64SparseMatrix, x: DoubleArray): DoubleArray {
    val y = DoubleArray(a.rows)
    for (j in 0 until a.cols) a.forEachInColumn(j) { i, v -> y[i] += v * x[j] }
    return y
}

internal fun assertSolvesAgreeWithReference(lapack: F64SparseLapack) {
    val rng = Random(20260815)
    for (n in intArrayOf(1, 2, 7, 23, 60)) {
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val host = lapack.factor(a)
        val portable = F64ReferenceSparseLinearAlgebra.factor(a)
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

internal fun assertAliasedDestinationSolves(lapack: F64SparseLapack) {
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

internal fun assertDeterminantAgreesWithReference(lapack: F64SparseLapack) {
    val rng = Random(20260817)
    for (n in intArrayOf(1, 3, 8)) {
        val a = sparseConformanceSystem(n, rng)
        val host = lapack.factor(a).determinant()
        val portable = F64ReferenceSparseLinearAlgebra.factor(a).determinant()
        // The comparison is relative because the values grow with n.
        assertTrue(abs(host / portable - 1.0) < 1e-9, "n=$n determinant disagreed: $host vs $portable")
    }
}

/** A host that cannot name the failing pivot must say so rather than invent a position. */
internal fun assertSingularIsReportedWithUnknownPosition(lapack: F64SparseLapack) {
    val rank1 = F64SparseMatrix.ofColumns(
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

/**
 * A degenerate matrix is answered portably: an empty one factors to a size-zero factorization and one of all
 * zeros is reported singular. On Kotlin/Native the reason is that `usePinned` has no address for an empty array.
 */
internal fun assertEmptyAndZeroMatricesTakeThePortablePath(lapack: F64SparseLapack) {
    val empty = lapack.factor(F64SparseMatrix.ofColumns(0, 0, emptyList()))
    assertEquals(0, empty.n, "an empty matrix factors to an empty factorization")
    val zeros = lapack.factor(F64SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList())))
    assertTrue(zeros.singular, "a matrix of zeros is singular")
}

/**
 * A host sparse factorization wins only its own half of the registry, so the sparse BLAS stays with the
 * reference. [n] sets the size of the system whose fill is reported.
 */
internal fun assertRegistersAsTheSparseLapackHalf(lapack: F64SparseLapack, n: Int) {
    withCleanBackends {
        registerBackend(lapack)
        assertEquals(lapack.name, koblas.sparseLapack.name, "${lapack.name} should win the sparse lapack half")
        assertEquals("reference", koblas.sparseBlas.name)

        val a = sparseConformanceSystem(n, Random(20260819))
        val f = koblas.factor(a)
        assertTrue(f.nnz >= n, "fill should at least cover the diagonals of L and U, got ${f.nnz}")
    }
}

/** Equilibration and a drop tolerance are koblas's own, so a host must hand those requests back. */
internal fun assertUnsupportedRequestsFallBack(
    lapack: F64SparseLapack,
    hostFactorization: (F64SparseFactorization) -> Boolean,
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
internal fun assertRepeatedFactorizationsSurvive(lapack: F64SparseLapack) {
    val rng = Random(20260820)
    val a = sparseConformanceSystem(120, rng)
    var checksum = 0.0
    repeat(300) {
        val f = lapack.factor(a)
        checksum += abs(f.solve(DoubleArray(120) { 1.0 })[0])
    }
    assertTrue(checksum > 0.0, "the loop should have produced solutions")
}

/**
 * The Control array both bindings build: UMFPACK's own defaults with iterative refinement off, so a solve is
 * the triangular solve against the factors and nothing more.
 *
 * Takes the two values rather than a binding, because the JVM reads them out of a `MemorySegment` and the
 * native side out of a `DoubleArray`. A null means no Control array was built, which is a failure in itself:
 * without one the solve reverts to UMFPACK's default of two refinement steps.
 */
internal fun assertControlArrayKeepsUmfpackDefaults(refinementSteps: Double?, pivotTolerance: Double?) {
    assertEquals(0.0, refinementSteps, "umfpack solves must not refine")
    assertEquals(0.1, pivotTolerance, "umfpack_di_defaults did not fill the control array")
}
