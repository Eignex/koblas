package com.eignex.koblas.sparse.host.klu

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
