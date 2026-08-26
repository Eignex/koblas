package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.sparse.F64BasisFactorizations
import com.eignex.koblas.sparse.F64RepeatedSparseLu
import com.eignex.koblas.sparse.basis.F64BasisSolvers
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import kotlin.test.Test
import kotlin.test.assertIs

/**
 * What a caller asks for a sparse backend by name for is the routines that backend carries outside the seam,
 * so the lookup has to hand back the type they sit on. Nothing here calls a library.
 */
class NamedSparseBackendTest {

    @Test
    fun `BASICLU by name carries its own basis factorization`() = withCleanBackends {
        registerBackend(BasicluSparseLu())

        val found = backendNamed("basiclu", F64Capabilities.basisFactorizations)

        assertIs<F64BasisFactorizations>(found)
    }

    @Test
    fun `KLU by name carries its own analysis reuse`() = withCleanBackends {
        registerBackend(KluSparseLu())

        val found = backendNamed("klu", F64Capabilities.repeatedSparseLu)

        assertIs<F64RepeatedSparseLu>(found)
    }

    @Test
    fun `HFactor by name carries the basis solver it is here for`() = withCleanBackends {
        registerBackend(HfactorSparseLu())

        val found = backendNamed("hfactor", F64Capabilities.basisSolvers)

        assertIs<F64BasisSolvers>(found)
    }
}
