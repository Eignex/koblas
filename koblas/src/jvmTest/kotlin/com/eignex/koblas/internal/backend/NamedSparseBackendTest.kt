package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import kotlin.test.Test
import kotlin.test.assertIs

/**
 * What a caller asks for a sparse backend by name for is the routines that backend carries outside the seam,
 * so the lookup has to hand back the type they sit on. Nothing here calls a library.
 */
class NamedSparseBackendTest {

    @Test
    fun `HFactor by name carries the basis solver it is here for`() = withCleanBackends {
        registerBackend(HfactorSparseLu())

        val found = backendNamed("hfactor", Capabilities.basisSolvers)

        assertIs<F64BasisSolvers>(found)
    }
}
