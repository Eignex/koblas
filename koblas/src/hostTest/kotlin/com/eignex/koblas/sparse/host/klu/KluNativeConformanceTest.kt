// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.sparse.*
import kotlin.random.Random
import kotlin.test.*

/** Checks the native KLU bindings against the reference implementation. */
class KluNativeConformanceTest {

    /** Required rather than skipped, since Kotlin/Native has no Assume and a skipped suite reads as green. */
    private val klu = KluSparseLu(KluConfig(factorizeMin = 0)).also {
        require(it.isAvailable) { "host SuiteSparse expected in the test environment" }
    }

    @Test
    fun `the binding reports what it is`() {
        assertEquals("klu", klu.name)
        assertEquals(HOST_BACKEND_PRIORITY + 1, klu.priority)
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() = assertSolvesAgreeWithReference(klu)

    @Test
    fun `an aliased destination still solves correctly`() = assertAliasedDestinationSolves(klu)

    @Test
    fun `the reciprocal pivot condition estimate is bounded`() = assertReciprocalPivotConditionEstimateIsBounded(klu)

    @Test
    fun `a singular matrix is reported rather than solved`() = assertSingularIsReportedWithUnknownPosition(klu)

    @Test
    fun `a refactorization on the same pattern keeps the factors`() {
        val rng = Random(20260831)
        val n = 10
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val first = klu.factor(a)
        val second = klu.refactor(first, a)

        assertSame(first, second, "the same pattern should have been refactorized in place")
        assertClose(
            F64ReferenceSparseLinearAlgebra.factor(a).solve(b),
            second.solve(b),
            "refactorized",
            tolerance = 1e-9,
        )
    }
}
