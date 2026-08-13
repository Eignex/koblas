// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.umfpack

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.assertAliasedDestinationSolves
import com.eignex.koblas.sparse.assertDeterminantAgreesWithReference
import com.eignex.koblas.sparse.assertRepeatedFactorizationsSurvive
import com.eignex.koblas.sparse.assertSingularIsReportedWithUnknownPosition
import com.eignex.koblas.sparse.assertSolvesAgreeWithReference
import com.eignex.koblas.sparse.assertUnsupportedRequestsFallBack
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Checks the native UMFPACK bindings against the reference implementation. */
class UmfpackNativeConformanceTest {

    /**
     * Required rather than skipped. Kotlin/Native has no Assume, and silently passing a test that exercised
     * nothing is worse than a failure with a reason on it.
     */
    private val umfpack: UmfpackSparseLapack = requireNotNull(
        UmfpackLoader.functions?.let { UmfpackSparseLapack(it) },
    ) { "host SuiteSparse expected in the test environment" }

    @Test
    fun `the binding resolves and registers`() {
        assertEquals("umfpack", umfpack.name)
        assertEquals(HOST_BACKEND_PRIORITY, umfpack.priority, "every koblas host binding registers at one priority")
        assertEquals("umfpack", koblas.sparseLapack.name, "discovery should have registered the backend")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() = assertSolvesAgreeWithReference(umfpack)

    @Test
    fun `an aliased destination still solves correctly`() = assertAliasedDestinationSolves(umfpack)

    @Test
    fun `the determinant agrees with the portable factorization`() = assertDeterminantAgreesWithReference(umfpack)

    @Test
    fun `a singular matrix is reported singular with an unknown position`() =
        assertSingularIsReportedWithUnknownPosition(umfpack)

    @Test
    fun `equilibrate and a drop tolerance fall back to the portable factorization`() =
        assertUnsupportedRequestsFallBack(umfpack) { it is UmfpackFactorization }

    /** `usePinned` has no address for an empty array, so these shapes take the portable path instead of UMFPACK. */
    @Test
    fun `empty and all-zero matrices take the portable path`() {
        val empty = umfpack.factor(SparseMatrix.ofColumns(0, 0, emptyList()))
        assertEquals(0, empty.n)
        val zeros = umfpack.factor(SparseMatrix.ofColumns(3, 3, listOf(emptyList(), emptyList(), emptyList())))
        assertTrue(zeros.singular, "a matrix of zeros is singular")
    }

    @Test
    fun `repeated factorizations do not exhaust native memory`() = assertRepeatedFactorizationsSurvive(umfpack)
}
