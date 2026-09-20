package com.eignex.koblas

import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.openBlas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * That the JVM default stays out of host libraries, which is a policy and not an accident of this host.
 *
 * Kotlin/Native's default composes an installed library into whole dense Level 2 and 3 calls. The JVM's does
 * not, and the reason is measured rather than stylistic: reaching a library from here copies every operand
 * into native memory first, so the call costs a pass over the data before any arithmetic, and the Vector API
 * kernels are ahead without it. A host binding remains explicitly callable for applications and comparisons.
 *
 * The consequence worth pinning is what it buys: ordinary JVM execution resolves no library, so whether one
 * is installed, whether it loads, and whether this process was given native access cannot affect it. These
 * cases assert that from inside a process that does have a library available, which is the case where a
 * composition would show up.
 */
class JvmDefaultEngineTest {
    @Test
    fun `the default engine resolves no host library even where one is installed`() {
        val installed = openBlas()

        assertEquals(null, koblas.vendor, "the JVM default resolved a host library")
        if (installed == null) {
            println("SKIPPED: no CBLAS library installed; the composition was ruled out structurally only")
        }
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `no built in jvm engine composes a library into a dense matrix call`() {
        val product = DenseCall(ORDER, ORDER, depth = ORDER)
        val engines = listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd)

        for (engine in engines) {
            val route = engine.denseRouteOf(DenseMatrixOperation.Gemm, product)

            assertEquals("portable-dense", route.scheduling, "${engine.name} composed a library")
            assertEquals(null, route.host, "${engine.name} named a vendor call")
            assertEquals(null, engine.vendor, "${engine.name} retained a binding")
        }
    }

    /**
     * The same call computed on all three arms, with a library present and none of them reaching it.
     *
     * A route says which schedule was chosen; this says the chosen one computes, which is the half a route
     * cannot cover.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `every jvm arm computes a dense product without any library`() {
        val a = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + it % 7 })
        val expected = DenseMatrix.zero(ORDER, ORDER)
        ReferenceBlas.gemm(1.0, a, false, a, false, 0.0, expected)

        for (engine in listOfNotNull(koblas, BuiltinEngines.scalar, BuiltinEngines.simd)) {
            val c = DenseMatrix.zero(ORDER, ORDER)
            engine.gemm(1.0, a, false, a, false, 0.0, c)

            assertClose(expected.values, c.values, "${engine.name} product", TIGHT_TOLERANCE)
        }
    }

    /**
     * An explicit binding stays reachable and stays transfer-inclusive.
     *
     * The transfer is not an implementation detail to hide: on this platform every operand is copied into
     * native memory to reach the library, and a measurement of the call includes that copy. The route says
     * so, which is what keeps a JVM vendor arm honest about what it is timing.
     */
    @Test
    fun `an explicit jvm host call names the transfer it pays for`() {
        val blas = openBlas() ?: return println(
            "SKIPPED: no CBLAS library installed; the explicit JVM binding was not exercised",
        )
        val a = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 1.0 + it % 5 })
        val c = DenseMatrix.zero(ORDER, ORDER)

        val route = blas.routeOf(
            BlasOperation.Gemm,
            matrices = listOf(a, a, c),
        )
        blas.gemm(1.0, a, false, a, false, 0.0, c)

        assertNotEquals(null, route.adapter, "the JVM binding reported no transfer")
        assertTrue(route.exactlyMeasurable, "an explicit host call is not admissible as a vendor measurement")
        assertTrue(c.values.any { it != 0.0 }, "the explicit binding computed nothing")
    }

    private companion object {
        /** Past the size at which Kotlin/Native's default would hand the same call to a library. */
        const val ORDER = 64
    }
}
