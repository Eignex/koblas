package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.ReferenceBackend
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.VectorKernels
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BackendAvailabilityTest {

    /** A half that reports itself unusable, which is what a binding does when its library did not resolve. */
    private class UnavailableHost(override val name: String) :
        Blas by ReferenceLinearAlgebra,
        Lapack by ReferenceLinearAlgebra {
        override val priority: Int get() = HOST_BACKEND_PRIORITY
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = false
        override val vectorKernels: VectorKernels get() = ReferenceLinearAlgebra.vectorKernels
    }

    @Test
    fun `the built-in backends report themselves available`() {
        assertTrue(ReferenceLinearAlgebra.isAvailable)
        assertTrue(ReferenceBackend().isAvailable)
        assertTrue(ReferenceSparseLinearAlgebra.isAvailable)
        assertTrue(ReferenceLinearAlgebra.vectorKernels.isAvailable, "the compiled-in kernels always run")
    }

    @Test
    fun `the default is available so a backend only opts out deliberately`() {
        val plain = object : Backend {
            override val name: String get() = "plain"
        }
        assertTrue(plain.isAvailable)
        assertFalse(UnavailableHost("absent").isAvailable)
    }

    @Test
    fun `a context is available only when every half is`() {
        assertTrue(koblas.isAvailable, "the installed context is assembled from resolved backends")
        val absent = UnavailableHost("absent")
        assertFalse(koblas.with(blas = absent).isAvailable, "one unavailable half makes the context so")
        assertFalse(koblas.with(lapack = absent).isAvailable)
    }
}
