package com.eignex.koblas.internal.backend

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.numeric.absoluteSum
import com.eignex.koblas.internal.numeric.euclideanNorm
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseKernels
import kotlin.test.*

class AccelerationTest {

    private class FakeHost(override val name: String) :
        F64Blas by F64ReferenceLinearAlgebra,
        F64Decompositions by F64ReferenceLinearAlgebra {
        override val priority: Int get() = 100
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
        override val kernels: F64Kernels get() = F64ReferenceLinearAlgebra.kernels
    }

    private class FakeKernels(override val name: String = "fakeblas") : F64Kernels {
        override val priority: Int get() = 100
        override fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
            var s = 0.0
            for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
            return s
        }
        override fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
            for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
        }
        override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
            for (i in 0 until len) v[vOff + i] *= alpha
        }
        override fun nrm2(v: DoubleArray, vOff: Int, len: Int): Double = euclideanNorm(v, vOff, len)
        override fun asum(v: DoubleArray, vOff: Int, len: Int): Double = absoluteSum(v, vOff, len)
    }

    @Test
    fun `an empty registry reports the context itself as portable`() = withCleanBackends {
        assertTrue(koblas.isPortable, "nothing is registered, so every half is koblas's own")
    }

    @Test
    fun `registered kernels carry their identity through the routing wrapper`() = withCleanBackends {
        registerBackend(FakeKernels())
        assertTrue(!koblas.isPortable, "a registered host half leaves the context accelerated")
        assertEquals(100, koblas.priority, "the context is as preferred as the best half in it")
    }

    @Test
    fun `an empty registry reports every slot as portable`() = withCleanBackends {
        assertEquals(BackendSlot.entries.toSet(), koblas.portableSlots)
        for (slot in BackendSlot.entries) {
            assertTrue(!koblas.isAccelerated(slot), "$slot claimed acceleration with nothing registered")
        }
    }

    @Test
    fun `registering a host backend accelerates exactly its own slots`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        assertTrue(koblas.isAccelerated(BackendSlot.F64Blas))
        assertTrue(koblas.isAccelerated(BackendSlot.F64Decompositions))
        assertEquals(
            setOf(
                BackendSlot.F64Kernels,
                BackendSlot.F64SparseKernels,
                BackendSlot.F64SparseBlas,
                BackendSlot.F64SparseLu,
            ),
            koblas.portableSlots,
        )
    }

    @Test
    fun `the compiled-in kernels do not count as acceleration`() = withCleanBackends {
        assertTrue(!koblas.isAccelerated(BackendSlot.F64Kernels), "the platform kernels are portable koblas")
        registerBackend(FakeKernels())
        assertTrue(koblas.isAccelerated(BackendSlot.F64Kernels), "a registered host half is acceleration")
    }

    @Test
    fun `requireAccelerated passes for the slots that are covered`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        koblas.requireAccelerated(BackendSlot.F64Blas, BackendSlot.F64Decompositions)
        koblas.requireAccelerated()
    }

    @Test
    fun `requireAccelerated names the slots that fell back and what filled them`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val failure = assertFailsWith<IllegalStateException> {
            koblas.requireAccelerated(BackendSlot.F64Blas, BackendSlot.F64SparseLu)
        }
        val message = failure.message!!
        assertTrue("F64SparseLu=reference" in message, "should name the slot and its backend: $message")
        assertTrue("F64Blas=" !in message, "should not name a slot that is accelerated: $message")
        assertTrue("backend=" in message, "should include the resolved summary: $message")
    }

    @Test
    fun `a context reports its own halves rather than the global registry`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val portable = koblas.with(blas = F64ReferenceLinearAlgebra, decompositions = F64ReferenceLinearAlgebra)
        assertTrue(!portable.isAccelerated(BackendSlot.F64Blas), "the context's own half is the portable one")
        assertTrue(koblas.isAccelerated(BackendSlot.F64Blas), "the registry is still accelerated")
        assertFailsWith<IllegalStateException> { portable.requireAccelerated(BackendSlot.F64Blas) }
    }

    @Test
    fun `backendFor returns the half filling each slot`() = withCleanBackends {
        val host = FakeHost("openblas")
        registerBackend(host)
        assertSame(host, koblas.backendFor(BackendSlot.F64Blas))
        assertSame(host, koblas.backendFor(BackendSlot.F64Decompositions))
        assertSame(koblas.kernels, koblas.backendFor(BackendSlot.F64Kernels))
        assertSame(koblas.sparseBlas, koblas.backendFor(BackendSlot.F64SparseBlas))
    }

    /** A sparse-kernel half at the default priority, which is what most registrations use. */
    private class PlainSparseKernels : F64SparseKernels {
        override val name: String get() = "plain-sparse"
        override fun dot(x: F64SparseVector, y: DoubleArray): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)
        override fun dot(x: F64SparseVector, y: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.dot(x, y)
        override fun axpy(y: DoubleArray, alpha: Double, x: F64SparseVector) =
            F64ReferenceSparseLinearAlgebra.axpy(y, alpha, x)
        override fun scatter(x: F64SparseVector, out: DoubleArray) = F64ReferenceSparseLinearAlgebra.scatter(x, out)
        override fun nrm2(x: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.nrm2(x)
        override fun asum(x: F64SparseVector): Double = F64ReferenceSparseLinearAlgebra.asum(x)
    }

    /**
     * A seam nothing was registered for has to accept the next offer whatever its priority, including after
     * some earlier test has cleaned the registry. Restoring by re-registering the halves that were resolved
     * cannot manage that: a slot nothing was registered for reads as its compiled-in fallback, so putting
     * that back leaves a priority-0 incumbent occupying the seam.
     *
     * Registered outside a [withCleanBackends] block on purpose. Inside one the leading reset would hide the
     * problem, which is what makes this trap easy to miss.
     */
    @Test
    fun `a default-priority sparse kernel wins after an earlier test cleaned the registry`() {
        withCleanBackends { /* the clean-and-restore path under test */ }
        try {
            registerBackend(PlainSparseKernels())
            assertEquals("plain-sparse", koblas.sparseKernels.name, "an occupied seam refused the offer")
        } finally {
            resetBackends()
            rediscoverBackends()
        }
    }
}
