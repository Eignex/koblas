package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.backend.BackendNames
import com.eignex.koblas.koblas
import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.*
import com.eignex.koblas.testutil.host.HostLibraryTest
import com.eignex.koblas.withCleanBackends
import com.eignex.koblas.withColumn
import org.junit.Assume
import org.junit.experimental.categories.Category
import java.io.File
import kotlin.random.Random
import kotlin.test.*

@Category(HostLibraryTest::class)
class BasicluConformanceTest {

    // Every assertion below is about the native binding, and these systems are small enough that the
    // stored-entry gate would hand them to the portable factorization instead.
    private val basiclu = BasicluSparseLu(BasicluConfig())
    private val equilibratingBasiclu = BasicluSparseLu(BasicluConfig(equilibrate = true))

    private fun requireBasiclu() {
        Assume.assumeTrue("BASICLU is not installed; basiclu conformance cannot run", basiclu.isAvailable)
    }

    @Test
    fun `the binding resolves on a machine with basiclu`() {
        requireBasiclu()
        assertEquals("basiclu", basiclu.name)
        assertEquals(HOST_BACKEND_PRIORITY + 2, basiclu.priority)
        assertTrue(basiclu.supportsBasisUpdates, "the binding exists for its updates")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() {
        requireBasiclu()
        assertSolvesAgreeWithReference(basiclu)
    }

    @Test
    fun `a backend with no native Cholesky answers with the portable one`() {
        requireBasiclu()
        assertCholeskyAgreesWithReference(basiclu)
    }

    @Test
    fun `an aliased destination still solves correctly`() {
        requireBasiclu()
        assertAliasedDestinationSolves(basiclu)
    }

    @Test
    fun `repeated solves declare a strict allocation contract`() {
        requireBasiclu()
        assertStrictNativeSolveAllocationContract(basiclu)
    }

    @Test
    fun `the reciprocal pivot condition estimate is bounded`() {
        requireBasiclu()
        assertReciprocalPivotConditionEstimateIsBounded(basiclu)
    }

    @Test
    fun `a singular matrix is reported rather than solved`() {
        requireBasiclu()
        assertSingularIsReportedWithUnknownPosition(basiclu)
    }

    @Test
    fun `a native factor closes deterministically`() {
        requireBasiclu()
        val factorization = basiclu.factor(sparseConformanceSystem(8, Random(20261003)))
        assertIs<BasicluFactorization>(factorization)

        assertNativeFactorCloseContract(factorization)
    }

    @Test
    fun `an updated basis solves as the basis it became in both directions`() {
        requireBasiclu()
        val rng = Random(20260825)
        val n = 16
        val basis = sparseConformanceSystem(n, rng)
        val entering = F64SparseVector.of(n, intArrayOf(0, 5, 11), doubleArrayOf(2.0, -3.0, n + 4.0))
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val updated = basiclu.factorBasis(basis).replaceColumn(5, entering)

        assertEquals(basis.withColumn(5, entering), updated.basis)
        for (transpose in booleanArrayOf(false, true)) {
            assertClose(
                F64ReferenceSparseLinearAlgebra.factor(updated.basis).solve(b, transpose),
                updated.solve(b, transpose),
                "one replacement transpose=$transpose",
                tolerance = 1e-9,
            )
        }
    }

    @Test
    fun `a chain of replacements keeps following the basis`() {
        requireBasiclu()
        val rng = Random(20260826)
        val n = 12
        var expected = sparseConformanceSystem(n, rng)
        var factorization = basiclu.factorBasis(expected)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        for (column in 0 until n) {
            val entering = F64SparseVector.of(
                n,
                intArrayOf(column, (column + 5) % n),
                doubleArrayOf(n + 6.0, 1.0),
            )
            expected = expected.withColumn(column, entering)
            factorization = factorization.replaceColumn(column, entering)
        }

        assertEquals(expected, factorization.basis)
        assertClose(
            F64ReferenceSparseLinearAlgebra.factor(expected).solve(b),
            factorization.solve(b),
            "a chain of $n replacements",
            tolerance = 1e-9,
        )
    }

    @Test
    fun `a basis BASICLU calls singular is reported rather than solved`() {
        requireBasiclu()
        val basis = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 1.0)))

        val factorization = basiclu.factorBasis(basis)

        assertTrue(factorization.singular, "two equal columns are a singular basis")
    }

    @Test
    fun `an equilibrated factorization stays native and still solves`() {
        requireBasiclu()
        assertNativeEquilibration(equilibratingBasiclu) { it is BasicluFactorization }
    }

    @Test
    fun `an equilibrated factorization agrees with the portable one in both directions`() {
        requireBasiclu()
        val rng = Random(20260827)
        val n = 14
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val host = equilibratingBasiclu.factor(a)

        for (transpose in booleanArrayOf(false, true)) {
            assertClose(
                F64ReferenceSparseDecompositions(equilibrate = true).factor(a).solve(b, transpose),
                host.solve(b, transpose),
                "equilibrated transpose=$transpose",
                tolerance = 1e-9,
            )
        }
    }

    @Test
    fun `an equilibrated solve leaves its right-hand side alone`() {
        requireBasiclu()
        val rng = Random(20260828)
        val n = 8
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val untouched = b.copyOf()

        equilibratingBasiclu.factor(a).solve(b)

        assertContentEquals(untouched, b, "the right-hand side was scaled in place")
    }

    /**
     * HiGHS vendors BASICLU built for 32-bit `lu_int` and exports the same symbol names, which is the one
     * provider that would answer this binding and compute nonsense. The path is where Debian keeps it.
     */
    @Test
    fun `a provider of the wrong integer width is refused`() {
        val highs = File("/usr/lib/x86_64-linux-gnu/libhighs.so.1")
        Assume.assumeTrue("no 32-bit BASICLU on this machine to be refused", highs.isFile)
        assertFalse(BasicluSparseLu(BasicluConfig(highs.path)).isAvailable)
    }

    /**
     * BASICLU factorizes simplex bases, and [com.eignex.koblas.internal.backend.BackendSlot.F64GeneralSparseLu]
     * declines an offer from a library specialised that way: what it has for ordinary LU is that
     * specialization's own factorization rather than a general one. So the half it wins is the basis one, and
     * general LU stays with whoever is general.
     */
    @Test
    fun `it registers as the basis half and leaves general LU to a general provider`() {
        requireBasiclu()

        withCleanBackends {
            registerBackend(basiclu)

            assertEquals(BackendNames.BASICLU, koblas.basisFactorizations.name)
            assertEquals(BackendNames.REFERENCE, koblas.generalSparseLu.name)
            assertEquals(BackendNames.REFERENCE, koblas.sparseBlas.name)
        }
    }
}
