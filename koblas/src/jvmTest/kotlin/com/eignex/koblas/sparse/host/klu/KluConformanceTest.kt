package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.sparse.*
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.random.Random
import kotlin.test.*

@Category(HostLibraryTest::class)
class KluConformanceTest {
    private val klu = KluSparseLu(KluConfig(factorizeMin = 0))

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
    fun `a symbolic analysis refactors compatible values`() {
        Assume.assumeTrue("KLU is not installed; conformance cannot run", klu.isAvailable)

        assertSymbolicAnalysisReuses(klu)
    }
}
