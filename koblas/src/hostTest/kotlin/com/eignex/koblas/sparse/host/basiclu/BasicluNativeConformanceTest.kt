// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.sparse.*
import com.eignex.koblas.withColumn
import kotlin.random.Random
import kotlin.test.*

/**
 * Checks the native BASICLU binding against the reference implementation. No distribution packages BASICLU,
 * so unlike the SuiteSparse suites this one cannot require the library: it asserts conformance where it is
 * installed and the portable fallback where it is not, which leaves no run passing on nothing.
 */
class BasicluNativeConformanceTest {

    private val basiclu = BasicluSparseLu(BasicluConfig(factorizeMin = 0))
    private val equilibratingBasiclu = BasicluSparseLu(BasicluConfig(factorizeMin = 0, equilibrate = true))

    @Test
    fun `the binding reports what it is`() {
        assertEquals("basiclu", basiclu.name)
        assertEquals(HOST_BACKEND_PRIORITY + 2, basiclu.priority)
        assertEquals(basiclu.isAvailable, basiclu.supportsBasisUpdates, "updates come with the library")
    }

    @Test
    fun `solutions agree with the portable factorization in both directions`() {
        if (!basiclu.isAvailable) return
        assertSolvesAgreeWithReference(basiclu)
    }

    @Test
    fun `an aliased destination still solves correctly`() {
        if (!basiclu.isAvailable) return
        assertAliasedDestinationSolves(basiclu)
    }

    @Test
    fun `repeated solves declare a strict allocation contract`() {
        if (!basiclu.isAvailable) return
        assertStrictNativeSolveAllocationContract(basiclu)
    }

    @Test
    fun `a native factor closes deterministically`() {
        if (!basiclu.isAvailable) return
        val factorization = basiclu.factor(sparseConformanceSystem(8, Random(20261009)))
        assertIs<BasicluFactorization>(factorization)

        assertNativeFactorCloseContract(factorization)
    }

    /** The replacement path where BASICLU is installed, the refactoring fallback where it is not. */
    @Test
    fun `a replaced basis column solves as the basis it became`() {
        val rng = Random(20260829)
        val n = 11
        val basis = sparseConformanceSystem(n, rng)
        val entering = F64SparseVector.of(n, intArrayOf(0, 4, 9), doubleArrayOf(2.0, -1.0, n + 5.0))
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val updated = basiclu.factorBasis(basis).replaceColumn(4, entering)

        assertEquals(basis.withColumn(4, entering), updated.basis)
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
    fun `an equilibrated factorization agrees with the portable one`() {
        if (!basiclu.isAvailable) return
        val rng = Random(20260830)
        val n = 9
        val a = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(
            F64ReferenceSparseDecompositions(equilibrate = true).factor(a).solve(b),
            equilibratingBasiclu.factor(a).solve(b),
            "equilibrated",
            tolerance = 1e-9,
        )
    }
}
