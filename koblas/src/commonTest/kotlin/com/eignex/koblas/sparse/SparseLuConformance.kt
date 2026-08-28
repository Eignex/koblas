package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
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
            assertClose(b, koblas.gemv(a, fromHost, transpose), "n=$n transpose=$transpose residual", tolerance = 1e-9)
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

/** A host factor declares and executes at least a size-independent-managed solve without hidden fallback. */
internal fun assertStrictNativeSolveAllocationContract(decompositions: F64SparseDecompositions) {
    val rng = Random(20260827)
    val n = 16
    val a = sparseConformanceSystem(n, rng)
    val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
    val factor = decompositions.factor(a)
    val out = DoubleArray(n)

    assertTrue(
        factor.solveAllocation(aliasing = false).supports(AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED),
        "${decompositions.name} did not declare its repeated-solve allocation contract",
    )
    factor.solveInto(
        b,
        out,
        allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
    )

    assertClose(
        F64ReferenceSparseLinearAlgebra.factor(a).solve(b),
        out,
        "strict native allocation solve",
        tolerance = 1e-9,
    )
}

/** A provider's block solve agrees column-for-column and preserves its in-place contract. */
internal fun assertBlockSolvesAgreeWithReference(decompositions: F64SparseDecompositions) {
    val rng = Random(20260830)
    val n = 12
    val a = sparseConformanceSystem(n, rng)
    val factor = decompositions.factor(a)
    val portable = F64ReferenceSparseLinearAlgebra.factor(a)
    val b = F64DenseMatrix(n, 4, DoubleArray(n * 4) { rng.nextDouble(-1.0, 1.0) })
    assertTrue(factor.nnz >= n)
    assertTrue(factor.rcond > 0.0)
    for (transpose in booleanArrayOf(false, true)) {
        val expected = portable.solve(b, transpose)
        val actual = factor.solve(b, transpose)
        assertClose(expected.data, actual.data, "block transpose=$transpose", tolerance = 1e-9)

        val aliased = F64DenseMatrix.wrap(n, b.cols, b.data.copyOf())
        factor.solveInto(aliased, aliased, transpose)
        assertClose(expected.data, aliased.data, "aliased block transpose=$transpose", tolerance = 1e-9)
    }
}

/** A factor with a native block ABI agrees with a portable factor and preserves its in-place contract. */
internal fun assertNativeBlockFactorSolvesAgreeWithReference(
    factor: F64SparseFactorization,
    portable: F64SparseFactorization,
) {
    val rng = Random(20260831)
    val b = F64DenseMatrix(factor.n, 4, DoubleArray(factor.n * 4) { rng.nextDouble(-1.0, 1.0) })
    for (transpose in booleanArrayOf(false, true)) {
        val expected = portable.solve(b, transpose)
        val actual = factor.solve(b, transpose)
        assertClose(expected.data, actual.data, "symmetric block transpose=$transpose", tolerance = 1e-9)

        val aliased = F64DenseMatrix.wrap(factor.n, b.cols, b.data.copyOf())
        factor.solveInto(aliased, aliased, transpose)
        assertClose(expected.data, aliased.data, "aliased symmetric block transpose=$transpose", tolerance = 1e-9)
    }
}

/** Strict solve contract for a native symmetric factor reached through a different semantic seam. */
internal fun assertStrictNativeSolveAllocationContract(factor: F64SparseFactorization, b: DoubleArray) {
    val expected = factor.solve(b)
    val out = DoubleArray(factor.n)
    assertTrue(
        factor.solveAllocation(aliasing = false).supports(AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED),
        "native factor did not declare its repeated-solve allocation contract",
    )

    factor.solveInto(
        b,
        out,
        allocationPolicy = AllocationPolicy.REQUIRE_NO_SIZE_DEPENDENT_MANAGED,
    )

    assertClose(expected, out, "strict native symmetric solve", tolerance = 1e-12)
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
    assertClose(b, koblas.gemv(a, equilibrated.solve(b)), "equilibrated residual", tolerance = 1e-9)
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
    when (factorization) {
        is F64SparseLuFactorization -> {
            assertFailsWith<IllegalStateException> { factorization.l }
            assertFailsWith<IllegalStateException> { factorization.u }
            assertFailsWith<IllegalStateException> { factorization.rowOrder }
            assertFailsWith<IllegalStateException> { factorization.columnOrder }
            assertFailsWith<IllegalStateException> { factorization.rowScaling }
            assertFailsWith<IllegalStateException> { factorization.offDiagonal }
        }

        is F64SparseCholeskyFactorization -> {
            assertFailsWith<IllegalStateException> { factorization.l }
            assertFailsWith<IllegalStateException> { factorization.order }
        }

        is F64SparseLdlFactorization -> {
            assertFailsWith<IllegalStateException> { factorization.l }
            assertFailsWith<IllegalStateException> { factorization.d }
            assertFailsWith<IllegalStateException> { factorization.order }
        }
    }
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
