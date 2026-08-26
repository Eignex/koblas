// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.sparse.*
import kotlinx.cinterop.*
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
    fun `shared options reach the effective KLU common block`() {
        val loader = KluLoader(
            KluConfig(
                libraryPath = null,
                options = KluOptions(
                    factorizeMin = 0,
                    equilibrate = true,
                    pivotTolerance = 0.25,
                    memoryGrowth = 1.5,
                    amdInitialMemoryFactor = 1.75,
                    initialMemoryFactor = 2.0,
                    maxBtfWork = 3.0,
                    useBtf = false,
                    ordering = KluOrdering.COLAMD,
                    equilibratedScaling = KluScaling.MAX,
                    haltIfSingular = true,
                ),
            ),
        )
        require(loader.available) { "host SuiteSparse expected in the test environment" }

        val common = loader.common(equilibrate = true)
        try {
            assertEquals(0.25, doubleAt(common, KLU_COMMON_TOL))
            assertEquals(1.5, doubleAt(common, KLU_COMMON_MEMGROW))
            assertEquals(1.75, doubleAt(common, KLU_COMMON_INITMEM_AMD))
            assertEquals(2.0, doubleAt(common, KLU_COMMON_INITMEM))
            assertEquals(3.0, doubleAt(common, KLU_COMMON_MAXWORK))
            assertEquals(0, intAt(common, KLU_COMMON_BTF))
            assertEquals(1, intAt(common, KLU_COMMON_ORDERING))
            assertEquals(KLU_SCALE_MAX, intAt(common, KLU_COMMON_SCALE))
            assertEquals(1, intAt(common, KLU_COMMON_HALT_IF_SINGULAR))
        } finally {
            nativeHeap.free(common)
        }
    }

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
