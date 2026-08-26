package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

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

internal fun assertSolvesAgreeWithReference(decompositions: F64SparseDecompositions) {
    val rng = Random(20260815)
    for (n in intArrayOf(1, 2, 7, 23, 60)) {
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val host = decompositions.factor(a)
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

internal fun assertAliasedDestinationSolves(decompositions: F64SparseDecompositions) {
    val rng = Random(20260816)
    val n = 12
    val a = sparseConformanceSystem(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    val f = decompositions.factor(a)
    val expected = f.solve(b)
    val aliased = b.copyOf()
    f.solveInto(aliased, aliased)
    assertClose(expected, aliased, "aliased destination", tolerance = 1e-12)
}

internal fun assertReciprocalPivotConditionEstimateIsBounded(decompositions: F64SparseDecompositions) {
    val identity = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
    assertEquals(1.0, decompositions.factor(identity).rcond)

    val estimate = decompositions.factor(sparseConformanceSystem(12, Random(20260824))).rcond
    assertTrue(estimate in 0.0..1.0, "reciprocal pivot condition estimate out of range: $estimate")
}

/** A host that cannot name the failing pivot must say so rather than invent a position. */
internal fun assertSingularIsReportedWithUnknownPosition(decompositions: F64SparseDecompositions) {
    val rank1 = F64SparseMatrix.ofColumns(
        2,
        2,
        listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 2.0, 1 to 4.0)),
    )
    val f = decompositions.factor(rank1)
    assertTrue(f.singular, "a rank-1 matrix should have been called singular")
    assertEquals(SINGULAR_POSITION_UNKNOWN, f.failedAt, "a host that cannot name the pivot must say so")
    assertEquals(0, f.nnz, "a singular factorization has no fill")
    assertEquals(0.0, f.rcond)
    assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 1.0)) }
}

/**
 * A host sparse factorization wins only its own half of the registry, so the sparse BLAS stays with the
 * reference. [n] sets the size of the system whose fill is reported.
 */
internal fun assertRegistersAsTheSparseLuHalf(decompositions: F64SparseDecompositions, n: Int) {
    withCleanBackends {
        registerBackend(decompositions)
        assertEquals(
            decompositions.name,
            koblas.sparseDecompositions.name,
            "${decompositions.name} should win the sparse decompositions half",
        )
        assertEquals("reference", koblas.sparseBlas.name)

        val a = sparseConformanceSystem(n, Random(20260819))
        val f = koblas.factor(a)
        assertTrue(f.nnz >= n, "fill should at least cover the diagonals of L and U, got ${f.nnz}")
    }
}

/**
 * Below its gate a host hands the factorization to the portable implementation, and at the gate it takes it
 * back. [gated] builds the binding with a chosen gate, since the gate is per-binding configuration. The two
 * gates straddle one matrix's stored-entry count, which is the quantity a sparse host gates on.
 */
internal fun assertFactorizeGateFallsBackToReference(
    gated: (factorizeMin: Int) -> F64SparseDecompositions,
    hostFactorization: (F64SparseFactorization) -> Boolean,
) {
    val a = sparseConformanceSystem(12, Random(20260901))
    val below = gated(a.nnz + 1).factor(a)
    assertTrue(!hostFactorization(below), "above the gate the portable factorization should answer")
    val at = gated(a.nnz).factor(a)
    assertTrue(hostFactorization(at), "at the gate the host factorization should answer")
}

/** A gated host still agrees with the reference on a problem it hands back. */
internal fun assertGatedFallbackStillSolves(gated: (factorizeMin: Int) -> F64SparseDecompositions) {
    val n = 7
    val rng = Random(20260902)
    val a = sparseConformanceSystem(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    val fromGated = gated(a.nnz + 1).factor(a).solve(b)
    val fromPortable = F64ReferenceSparseLinearAlgebra.factor(a).solve(b)
    assertClose(fromPortable, fromGated, "a gated host below its gate", tolerance = 1e-12)
}

/** A backend set to equilibrate scales natively and still solves the system it was given. */
internal fun assertNativeEquilibration(
    decompositions: F64SparseDecompositions,
    hostFactorization: (F64SparseFactorization) -> Boolean,
) {
    val rng = Random(20260818)
    val a = sparseConformanceSystem(6, rng)
    val b = DoubleArray(6) { rng.nextDouble(-1.0, 1.0) }
    val equilibrated = decompositions.factor(a)
    assertTrue(hostFactorization(equilibrated), "native equilibration must retain the host factorization")
    assertClose(b, a.gemv(equilibrated.solve(b)), "equilibrated residual", tolerance = 1e-9)
}

/** Native handles are freed per factorization, so a long loop must not grow without bound. */
internal fun assertRepeatedFactorizationsSurvive(decompositions: F64SparseDecompositions) {
    val rng = Random(20260820)
    val a = sparseConformanceSystem(120, rng)
    var checksum = 0.0
    repeat(300) {
        val f = decompositions.factor(a)
        checksum += abs(f.solve(DoubleArray(120) { 1.0 })[0])
    }
    assertTrue(checksum > 0.0, "the loop should have produced solutions")
}

/** A native factor releases idempotently, keeps stable facts, and rejects every resource-backed operation. */
internal fun assertNativeFactorCloseContract(factorization: F64SparseFactorization) {
    val n = factorization.n
    val failedAt = factorization.failedAt

    factorization.close()
    factorization.close()

    assertEquals(n, factorization.n)
    assertEquals(failedAt, factorization.failedAt)
    assertFailsWith<IllegalStateException> { factorization.nnz }
    assertFailsWith<IllegalStateException> { factorization.rcond }
    assertFailsWith<IllegalStateException> { factorization.solve(DoubleArray(n)) }
}

/** [AutoCloseable.use] closes a native factor when its body returns and when its body throws. */
internal fun assertNativeFactorUseContract(factory: () -> F64SparseFactorization) {
    val returned = factory()
    returned.use { assertTrue(it.nnz >= it.n) }
    assertFailsWith<IllegalStateException> { returned.nnz }

    val failed = factory()
    assertFailsWith<IllegalArgumentException> {
        failed.use { throw IllegalArgumentException("test failure") }
    }
    assertFailsWith<IllegalStateException> { failed.nnz }
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
