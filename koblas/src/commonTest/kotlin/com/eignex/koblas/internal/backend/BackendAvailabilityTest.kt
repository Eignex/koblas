package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.ReferenceSparseLinearAlgebra
import kotlin.test.*

class BackendAvailabilityTest {

    /** A half that reports itself unusable, which is what a binding does when its library did not resolve. */
    private class UnavailableHost(override val name: String) : Blas by ReferenceBlas {
        override val priority: Int get() = HOST_BACKEND_PRIORITY
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = false
        override val unavailableReason: String? get() = "$name did not resolve"
        override val kernels: Kernels get() = ReferenceBlas.kernels
    }

    @Test
    fun `the built-in backends report themselves available`() {
        assertTrue(ReferenceBlas.isAvailable)
        assertTrue(ReferenceBackend().isAvailable)
        assertTrue(ReferenceSparseLinearAlgebra.isAvailable)
        assertTrue(ReferenceBlas.kernels.isAvailable, "the compiled-in kernels always run")
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
    fun `the default reports no reason so a backend only explains itself deliberately`() {
        val plain = object : Backend {
            override val name: String get() = "plain"
        }
        assertNull(plain.unavailableReason)
        assertNull(ReferenceBlas.unavailableReason, "koblas's own halves always run")
    }

    @Test
    fun `an unavailable half reports its reason through the context status`() {
        val absent = UnavailableHost("absent")
        val context = koblas.with(blas = absent)

        val status = context.status[BackendRole.DENSE_BLAS]

        assertFalse(status.available)
        assertEquals("absent did not resolve", status.unavailableReason, "read without knowing the binding's type")
    }

    @Test
    fun `a context reports the first half that says why it cannot run`() {
        assertNull(koblas.unavailableReason, "the installed context is assembled from resolved backends")
        assertEquals("absent did not resolve", koblas.with(blas = UnavailableHost("absent")).unavailableReason)
    }

    @Test
    fun `a context is available only when every half is`() {
        assertTrue(koblas.isAvailable, "the installed context is assembled from resolved backends")
        assertFalse(koblas.with(blas = UnavailableHost("absent")).isAvailable)
    }
}
