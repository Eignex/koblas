package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.BackendExecution
import com.eignex.koblas.BackendRouteReason
import com.eignex.koblas.DispatchMetric
import com.eignex.koblas.F64RouteQuery
import com.eignex.koblas.testutil.host.HostLibraryTest
import org.junit.Assume
import org.junit.experimental.categories.Category
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

/**
 * The rank threshold the blocked host update applies, read through the route it reports.
 *
 * Carries the host-library category because an unavailable binding answers `BACKEND_UNAVAILABLE` before it
 * reaches the threshold, which is the right answer and not the one under test.
 */
@Category(HostLibraryTest::class)
class CholeskyUpdateRoutingTest {
    private val decompositions = F64Backends().decompositions

    private fun requireLapacke() {
        Assume.assumeTrue("LAPACKE is not installed; the route cannot reach its threshold", decompositions.isAvailable)
    }

    @Test
    fun `an update of too few vectors reports the rank that kept it portable`() {
        requireLapacke()

        val route = decompositions.route(F64RouteQuery.CholeskyRankUpdate(order = 1024, rank = 16))

        assertNotNull(route)
        assertEquals(BackendRouteReason.BELOW_THRESHOLD, route.reason)
        assertEquals(BackendExecution.PORTABLE, route.execution)
        val gate = assertNotNull(route.gate)
        assertEquals(DispatchMetric.DIMENSION, gate.metric)
        assertEquals(16L, gate.actual)
        assertEquals(64L, gate.minimum, "the measured minimum rank for the blocked path")
    }

    @Test
    fun `an update of enough vectors routes to the host`() {
        requireLapacke()

        val route = decompositions.route(F64RouteQuery.CholeskyRankUpdate(order = 1024, rank = 64))

        assertNotNull(route)
        assertEquals(BackendRouteReason.NATIVE_ROUTE, route.reason)
        assertEquals(BackendExecution.NATIVE, route.execution)
        assertNull(route.gate)
    }
}
