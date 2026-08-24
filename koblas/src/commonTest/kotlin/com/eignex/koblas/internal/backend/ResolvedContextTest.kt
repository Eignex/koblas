package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Decompositions
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The context the seams resolve to is assembled on demand and cached against a change count. A registration
 * into one half therefore has to invalidate a context cached from an earlier read, and must not cost the
 * registrations already in the other halves.
 */
class ResolvedContextTest {

    private class BlasHalf(override val name: String) : F64Blas by F64ReferenceLinearAlgebra {
        override val priority: Int get() = 40
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
    }

    private class LapackHalf(override val name: String) : F64Decompositions by F64ReferenceLinearAlgebra {
        override val priority: Int get() = 40
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
    }

    @Test
    fun `a registration is visible after the context has already been read`() = withCleanBackends {
        registerBackend(BlasHalf("someblas"))
        assertEquals("someblas", koblas.blas.name)
        // Reading it above caches a context. The next registration has to be seen through that cache.
        registerBackend(LapackHalf("somelapack"))
        assertEquals("somelapack", koblas.decompositions.name, "the cached context hid a later registration")
        assertEquals("someblas", koblas.blas.name, "rebuilding the context dropped the earlier half")
    }

    @Test
    fun `clearing the registry is visible after the context has already been read`() = withCleanBackends {
        registerBackend(BlasHalf("someblas"))
        assertEquals("someblas", koblas.blas.name)
        resetBackends()
        assertEquals("reference", koblas.blas.name, "the cached context outlived the registration")
    }
}
