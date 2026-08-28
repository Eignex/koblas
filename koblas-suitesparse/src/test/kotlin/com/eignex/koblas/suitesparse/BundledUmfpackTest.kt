package com.eignex.koblas.suitesparse

import com.eignex.koblas.*
import com.eignex.koblas.sparse.host.umfpack.*
import kotlin.test.*
import kotlin.test.Test

class BundledUmfpackTest {
    @Test
    fun `loads bundled OpenBLAS before UMFPACK`() {
        assertTrue(BundledUmfpack().isAvailable)
    }

    /** A caller reaching this library by name gets its own routines only if the two share a type. */
    @Test
    fun `the bundled UMFPACK is the binding rather than a wrapper around it`() {
        assertIs<UmfpackSparseLu>(BundledUmfpack(), "the bundled providers all answer as the type their binding is")
    }

    @Test
    fun `shared options control bundled UMFPACK diagnostics`() {
        val backend = BundledUmfpack(
            UmfpackOptions(
                iterativeRefinementSteps = 3,
                pivotTolerance = 0.2,
                scaling = UmfpackScaling.MAX,
            ),
        )

        val route = assertNotNull(backend.route(F64RouteQuery.SparseLu(storedEntries = 16)))

        assertEquals(BackendExecution.NATIVE, route.execution)
        assertEquals("3", backend.backendMetadata.options["iterativeRefinementSteps"])
        assertEquals("0.2", backend.backendMetadata.options["pivotTolerance"])
        assertEquals("MAX", backend.backendMetadata.options["scaling"])
    }
}
