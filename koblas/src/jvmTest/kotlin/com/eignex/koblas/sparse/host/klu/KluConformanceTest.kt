package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.*

@Category(HostLibraryTest::class)
class KluConformanceTest {
    private val klu = KluSparseLu(KluConfig())

    @Test
    fun `a native factor closes deterministically`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)
        val factorization = klu.factor(sparseConformanceSystem(8, Random(20261005)))
        assertIs<KluFactorization>(factorization)

        assertNativeFactorCloseContract(factorization)
    }

    @Test
    fun `repeated solves declare a strict allocation contract`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)
        assertStrictNativeSolveAllocationContract(klu)
    }

    @Test
    fun `multiple right hand sides use the KLU block solve`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)
        assertBlockSolvesAgreeWithReference(klu)
    }

    @Test
    fun `a symbolic analysis refactors compatible values`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)

        assertSymbolicAnalysisReuses(klu)
    }

    @Test
    fun `refactoring onto a different order factors afresh rather than raising`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)
        // A mismatched order is reported the way a mismatched pattern of the same order already was, so a
        // caller reusing an analysis for the wrong matrix gets one answer either way. The JVM binding used
        // to raise here while the native one re-factored.
        val first = klu.factor(sparseConformanceSystem(6, Random(20260906)))
        val different = sparseConformanceSystem(8, Random(20260907))

        val refactored = klu.refactor(first, different)

        assertEquals(8, refactored.n, "refactor onto a different order should have factored the new matrix")
        val b = DoubleArray(8) { 1.0 }
        assertClose(
            F64ReferenceSparseLinearAlgebra.gemv(different, refactored.solve(b)),
            b,
            "refactored solve",
            tolerance = 1e-8,
        )
    }

    @Test
    fun `a structurally singular matrix factors as singular every time`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)
        // Column 1 is empty, so no pivot exists for it. This is the path that returns before the factor
        // is handed over, and it runs often enough in a circuit or simplex loop that anything it fails to
        // release accumulates.
        val singular = F64SparseMatrix.ofColumns(
            3,
            3,
            listOf(listOf(0 to 1.0), emptyList(), listOf(2 to 1.0)),
        )

        repeat(64) {
            val factorization = klu.factor(singular)

            assertTrue(factorization.singular, "a matrix with an empty column factored as non-singular")
        }
    }
}
