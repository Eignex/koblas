package com.eignex.koblas.internal.backend

import com.eignex.koblas.registerBackend
import com.eignex.koblas.sparse.host.basiclu.BasicluSparseLu
import com.eignex.koblas.sparse.host.hfactor.HfactorSparseLu
import com.eignex.koblas.sparse.host.klu.KluSparseLu
import com.eignex.koblas.sparseLuNamed
import com.eignex.koblas.withCleanBackends
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

        val found = sparseLuNamed("basiclu")

        assertIs<BasicluSparseLu>(found, "factorBasis lives here and nowhere else")
    }

    @Test
    fun `KLU by name carries its own analysis reuse`() = withCleanBackends {
        registerBackend(KluSparseLu())

        val found = sparseLuNamed("klu")

        assertIs<KluSparseLu>(found, "refactor lives here and nowhere else")
    }

    @Test
    fun `HFactor by name carries the basis solver it is here for`() = withCleanBackends {
        registerBackend(HfactorSparseLu())

        val found = sparseLuNamed("hfactor")

        assertIs<HfactorSparseLu>(found)
    }
}
