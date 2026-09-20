package com.eignex.koblas

import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.sparse.SparseOperation
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertSame

/**
 * Which Level 1 arm the default engine selected, which nothing else here would notice.
 *
 * Each platform prefers a different one: the Vector API kernels on the JVM, where reaching a foreign library
 * copies both operands first, and the vendor on Kotlin/Native, which pins them instead and has no Vector API.
 * Every other test asks only for a correct answer, and both arms give that, so a default that quietly kept
 * running portable Kotlin everywhere would pass all of them.
 */
class PlatformEngineTest {
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `dense and sparse vectors expose the same route contract`() {
        val engine = BuiltinEngines.scalar
        val routes = listOf(
            engine.routeOf(DenseOperation.Dot, 16),
            engine.routeOf(SparseOperation.DotDense, 16),
        )

        for (route in routes) {
            assertEquals(RouteKind.Direct, route.kind)
            assertEquals("scalar", route.implementation)
            assertEquals(true, route.exactlyMeasurable)
        }
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `exact built in engines do not resolve or execute a host library`() {
        val call = DenseCall(WIDE, WIDE)
        // A product route is a question about the shared dimension too, so the product call carries one.
        // Matrix routes visit cache blocks, so the Level 1 crossover fixture would enumerate millions of
        // blocks here. This bounded product is already large enough to exercise packed execution.
        val product = DenseCall(256, 256, depth = 256)

        assertEquals(null, BuiltinEngines.scalar.vendor)
        assertEquals(
            "portable-dense",
            BuiltinEngines.scalar.routeOf(DenseMatrixOperation.Gemm, product).scheduling,
        )
        assertEquals(
            "scalar-panel/multi-dot",
            BuiltinEngines.scalar.routeOf(DenseMatrixOperation.GemvTransposed, call).components.single(),
        )
        BuiltinEngines.simd?.let {
            assertEquals(null, it.vendor)
            assertEquals("portable-dense", it.routeOf(DenseMatrixOperation.Gemm, product).scheduling)
        }
    }

    /**
     * The JVM default shares the exact SIMD engine, so benchmark selection and ordinary calls agree.
     *
     * Without the Vector API the owned matrix backends are portable. Kotlin/Native can still compose
     * whole host calls over those backends; host routing is checked separately.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `the platform selects vector matrix backends where available`() {
        val vector = BuiltinEngines.simd

        if (vector == null) {
            // These are the owned matrix backends; Native whole-call host routing is independent of them.
            assertEquals("scalar-panel", koblas.panelKernels.name)
            assertEquals(BuiltinEngines.scalar.productKernels.name, koblas.productKernels.name)
            assertEquals(BuiltinEngines.scalar.triangularKernels.name, koblas.triangularKernels.name)
            return
        }
        assertSame(vector, koblas, "the platform default is a second composition rather than the measured arm")
        assertNotEquals(
            BuiltinEngines.scalar.panelKernels.name,
            koblas.panelKernels.name,
            "an available vector panel backend was not selected",
        )
        assertNotEquals(BuiltinEngines.scalar.productKernels.name, koblas.productKernels.name)
        assertNotEquals(BuiltinEngines.scalar.triangularKernels.name, koblas.triangularKernels.name)
    }

    /**
     * The panel backend is the platform's, and a matrix call says which of its bodies it reaches.
     *
     * A selection is not an execution: the Vector API backend hands a panel shorter than one lane block to the
     * portable body, and this asks the route rather than the engine's name.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `a wide panel reaches the vector body and a short one does not`() {
        val simd = BuiltinEngines.simd ?: return println(
            "SKIPPED: no Vector API engine on this host; the panel bodies a route names were not compared",
        )
        val work = DenseMatrixOperation.GemvTransposed

        val wide = simd.routeOf(work, DenseCall(WIDE, 4)).components.single()
        val short = simd.routeOf(work, DenseCall(1, 4)).components.single()

        assertNotEquals(short, wide, "a wide panel and a single row named the same body")
        assertEquals("scalar-panel/multi-dot", short)
    }

    /**
     * Asks the arm itself, not what the host happens to have installed.
     *
     * A vendor being present does not make the JVM's Level 1 arm accelerated: the JVM deliberately never
     * routes Level 1 to a library, so under `-Pkoblas.noSimd=true` on a host with oneMKL the engine is the
     * portable one while a vendor exists. The kernels' own name is what says which arm was selected.
     */
    @Test
    fun `a wide call reaches whichever accelerated arm this platform has`() {
        if (koblas.vectorKernels.name == PORTABLE) {
            println("SKIPPED: this platform selected the portable kernels; there is no accelerated arm to reach")
            return
        }

        val wide = koblas.routeOf(DenseOperation.Dot, WIDE).implementation

        assertNotEquals(PORTABLE, wide, "a wide dot stayed on the portable kernels")
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(PORTABLE, koblas.routeOf(DenseOperation.Dot, 1).implementation)
    }

    /**
     * `sum` is not a BLAS routine, so no library exports one to reach.
     *
     * It still vectorises where there are lanes to use, which is why this says where it cannot go rather than
     * where it must stay.
     */
    @Test
    fun `sum never names a vendor because none exports it`() {
        val vendor = koblas.vendor ?: return
        val vendorArm = "${vendor.vendor.vendorName.lowercase()}-level1"

        assertNotEquals(vendorArm, koblas.routeOf(DenseOperation.Sum, WIDE).implementation)
    }

    /**
     * The measured crossovers differ by operation, and the routing has to differ with them.
     *
     * One width is enough to catch the mistake this guards against, which is a single constant creeping back:
     * at 96 a dot has crossed and a norm has not. `dnrm2` rescales for overflow safety per element where the
     * portable kernel tries the plain sum of squares first, and `idamax` costs sixty nanoseconds or more
     * before it looks at anything, so both are repaid later than the rest. Only the Native arm routes Level 1
     * to a library at all, so everywhere else this says so rather than asserting the JVM into the same shape.
     */
    @Test
    fun `each operation crosses to the vendor at its own measured width`() {
        val vendor = koblas.vendor ?: return skipped("no library on this host")
        val arm = "${vendor.vendor.vendorName.lowercase()}-level1"
        if (koblas.vectorKernels.name != arm) return skipped("this platform keeps Level 1 off the library")

        assertEquals(PORTABLE, koblas.routeOf(DenseOperation.Dot, 32).implementation, "a dot at 32 is still the loop's")
        assertEquals(arm, koblas.routeOf(DenseOperation.Dot, 96).implementation, "a dot at 96 has crossed")
        assertEquals(PORTABLE, koblas.routeOf(DenseOperation.Nrm2, 96).implementation, "the robust norm crosses later")
        assertEquals(PORTABLE, koblas.routeOf(DenseOperation.Iamax, 96).implementation, "and so does the index search")
        assertEquals(arm, koblas.routeOf(DenseOperation.Nrm2, 128).implementation, "both have crossed by 128")
        assertEquals(arm, koblas.routeOf(DenseOperation.Iamax, 128).implementation)
    }

    private fun skipped(why: String) {
        println("SKIPPED: $why; the per-operation crossovers were not exercised here")
    }

    private companion object {
        const val PORTABLE = "scalar"

        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16
    }
}
