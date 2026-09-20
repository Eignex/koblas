// Backtick test names are this repository's convention, and detekt's default exclusions for them cover the
// standard test paths but not `nativeTest`. Stated here rather than changed in the shared lint configuration,
// which is the convention plugin's. Nothing else is relaxed: every case below carries its own documentation.
@file:Suppress("FunctionNaming")

package com.eignex.koblas

import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.sparse.SPARSE_SCHEDULING
import com.eignex.koblas.vendor.RouteKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * What this platform's default actually does with an installed library, which nothing common can ask.
 *
 * Kotlin/Native is the one platform whose default composes a host library into ordinary dense Level 2 and 3
 * calls, because its own arithmetic there is scalar and stays scalar. Everything else in the suite would pass
 * with the composition missing, since the portable schedule computes the same answers; these cases read the
 * route instead, so a default that quietly stopped reaching the library fails here.
 *
 * Every case says what it could not check where no library is installed, rather than passing silently. That
 * is the same host this library is expected to work on with nothing installed at all, and the portable
 * assertions below run there either way.
 */
class NativeDefaultEngineTest {
    private fun skipped(what: String) {
        println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
    }

    /**
     * The whole call goes to the library, and the route names which library, which symbol and which binary.
     *
     * An engine holding a binding is not evidence that a call reached it, so this asks the route for the call
     * it is about to describe and checks the identity against the binding the engine actually holds.
     */
    @Test
    fun `a product past the policy's size is one whole vendor call and says which`() {
        val vendor = koblas.vendor ?: return skipped("the composed dense route")

        val route = koblas.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = ORDER))

        assertEquals(HOST_SCHEDULING, route.scheduling, "the default did not compose the installed library")
        assertEquals(RouteKind.Direct, route.kind)
        assertEquals(vendor.vendor, route.host?.vendor, "the route names a library other than the engine's")
        assertEquals("cblas_dgemm", route.host?.entryPoint)
        assertTrue(vendor.libraryPath in (route.reason ?: ""), "the route does not name the resolved binary")
        assertTrue(vendor.version in (route.reason ?: ""), "the route does not name the library's version")
        assertTrue(
            vendor.threadEvidence.label in (route.reason ?: ""),
            "the route does not carry the library's threading evidence",
        )
    }

    /** Below the policy's size the same operation keeps this library's schedule, library installed or not. */
    @Test
    fun `a small call keeps the portable schedule beside the library`() {
        if (koblas.vendor == null) return skipped("the fallback below the policy's size")

        val route = koblas.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(SMALL, SMALL, depth = SMALL))

        assertNotEquals(HOST_SCHEDULING, route.scheduling, "a small product was sent to the library")
        assertEquals(null, route.host)
    }

    /**
     * A retained panel is grouped for this library's own register tile, which a library cannot read.
     *
     * Deliberately over the size at which the ordinary product goes across, so what settles it is the layout
     * and not the extents.
     */
    @Test
    fun `a product over retained panels stays this library's own however large it is`() {
        if (koblas.vendor == null) return skipped("the retained-layout fallback")
        val call = DenseCall(ORDER, ORDER, depth = ORDER)

        for (operation in PACKED) {
            val route = koblas.denseRouteOf(operation, call)

            assertNotEquals(HOST_SCHEDULING, route.scheduling, "$operation was labelled a host call")
            assertEquals(null, route.host, "$operation carried a vendor call")
        }
    }

    /**
     * A portable dense route can still reach the library, because this platform's Level 1 arm is the library.
     *
     * So `host == null` on a route means no single vendor entry point served the whole call, and not that no
     * library ran. The matrix-vector product below is too small to be handed over whole, and its destination
     * scaling is long enough to cross the Level 1 crossover, so the route is the portable schedule with a
     * vendor leaf named in its components. A reader who took the absent whole-call route for the absence of
     * the library would have this one backwards.
     */
    @Test
    fun `a portable route still names the vendor level one leaf it reaches`() {
        val vendor = koblas.vendor ?: return skipped("the mixed portable route")
        val arm = "${vendor.vendor.vendorName.lowercase()}-level1"
        if (koblas.vectorKernels.name != arm) return skipped("this platform keeps Level 1 off the library")
        val call = DenseCall(LONG_COLUMN, FEW_COLUMNS, beta = 0.5)

        val route = koblas.denseRouteOf(DenseMatrixOperation.Gemv, call)

        assertNotEquals(HOST_SCHEDULING, route.scheduling, "a small matrix-vector product was handed over")
        assertEquals(null, route.host, "a call this library scheduled carried a whole-call vendor route")
        assertTrue(
            route.components.any { it.startsWith(arm) },
            "the portable route did not name the vendor Level 1 leaf it reaches: ${route.components}",
        )
    }

    /** Sparse has no bound host entry point at all, so it is this library's scheduling on every engine. */
    @Test
    fun `sparse scheduling is unaffected by an installed library`() {
        assertEquals(SPARSE_SCHEDULING, koblas.sparseImplementation)
    }

    /**
     * Portability without a library, which is the property the whole composition must not cost.
     *
     * The exact portable engine is the floor: it resolves nothing, and every level computes there. Asserting
     * it here as well as in the common suite is what says the Native default's composition did not reach into
     * it.
     */
    @OptIn(KoblasEngineApi::class)
    @Test
    fun `the exact portable engine computes every level without a library`() {
        val engine = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(SMALL, SMALL, DoubleArray(SMALL * SMALL) { 1.0 + it })
        val c = DenseMatrix.zero(SMALL, SMALL)

        engine.gemm(1.0, a, false, a, false, 0.0, c)
        val route = engine.denseRouteOf(DenseMatrixOperation.Gemm, DenseCall(ORDER, ORDER, depth = ORDER))

        assertEquals(null, engine.vendor, "the exact portable engine resolved a host library")
        assertEquals("portable-dense", route.scheduling)
        assertTrue(c.values.any { it != 0.0 }, "the portable product computed nothing")
    }

    private companion object {
        /** Past the fixed policy's size as a cubic product, so the ordinary route goes to the library. */
        const val ORDER = 64

        /** Well under it as a cubic product, so the same operation stays portable. */
        const val SMALL = 4

        /** Past the Level 1 crossover, so the destination scaling of a small call reaches the library. */
        const val LONG_COLUMN = 256

        /** With [LONG_COLUMN], under the whole-call policy's size, so the call itself stays here. */
        const val FEW_COLUMNS = 2

        /** The name a composed host route carries, restated here rather than imported from production. */
        const val HOST_SCHEDULING = "host-dense"

        val PACKED = listOf(
            DenseMatrixOperation.GemmPacked,
            DenseMatrixOperation.GemmPackedLeft,
            DenseMatrixOperation.GemmPackedRight,
        )
    }
}
