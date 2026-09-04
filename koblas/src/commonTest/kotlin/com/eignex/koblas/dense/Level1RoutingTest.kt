package com.eignex.koblas.dense

import com.eignex.koblas.BackendExecution
import com.eignex.koblas.BackendRole
import com.eignex.koblas.BackendRouteReason
import com.eignex.koblas.DispatchGate
import com.eignex.koblas.DispatchMetric
import com.eignex.koblas.F64ContextBuilder
import com.eignex.koblas.F64Level1Routine
import com.eignex.koblas.F64RouteQuery
import com.eignex.koblas.route
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The level-1 crossovers, read through [route] rather than timed. A context with a host registered still
 * runs a short vector on the compiled-in kernels, and this is how a caller can tell before making the call.
 */
class Level1RoutingTest {

    /** Stands in for a registered host: what it computes does not matter, only that it is not portable. */
    private class FakeHost : F64Kernels by F64PlatformKernels {
        override val name: String get() = "fake-host"
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
        override val priority: Int get() = 100
    }

    private fun hostedContext() = F64ContextBuilder()
        .withBackend(BackendRole.DENSE_KERNELS, F64RoutedKernels(FakeHost()))
        .resolve()

    /** The eight gated routines share one measured crossover, and the route reports it for each. */
    @Test
    fun `a run below the crossover reports the gate that kept it portable`() {
        val context = hostedContext()

        for (routine in F64Level1Routine.entries - F64Level1Routine.ROTMG) {
            val route = context.route(F64RouteQuery.Level1(routine, length = 63))

            assertEquals(BackendRouteReason.BELOW_THRESHOLD, route.reason, "$routine at 63")
            assertEquals(BackendExecution.PORTABLE, route.execution, "$routine at 63")
            assertEquals(DispatchGate(DispatchMetric.VECTOR_LENGTH, 63, 64), route.gate, "$routine at 63")
        }
    }

    @Test
    fun `a run at the crossover routes to the host with nothing to report`() {
        val context = hostedContext()

        for (routine in F64Level1Routine.entries - F64Level1Routine.ROTMG) {
            val route = context.route(F64RouteQuery.Level1(routine, length = 64))

            assertEquals(BackendRouteReason.NATIVE_ROUTE, route.reason, "$routine at 64")
            assertEquals(BackendExecution.NATIVE, route.execution, "$routine at 64")
            assertNull(route.gate, "$routine at 64")
        }
    }

    /** Generating a rotation takes four scalars, so there is no length for a gate to compare. */
    @Test
    fun `the ungated routine routes to the host at any length`() {
        val context = hostedContext()

        val route = context.route(F64RouteQuery.Level1(F64Level1Routine.ROTMG, length = 1))

        assertEquals(BackendRouteReason.NATIVE_ROUTE, route.reason)
        assertNull(route.gate)
    }

    /** With nothing registered the half is koblas's own, which the context answers without asking. */
    @Test
    fun `a portable context reports no threshold at all`() {
        val context = F64ContextBuilder().resolve()

        val route = context.route(F64RouteQuery.Level1(F64Level1Routine.DOT, length = 1))

        assertEquals(BackendRouteReason.SELECTED_PORTABLE, route.reason)
        assertNull(route.gate)
    }
}
