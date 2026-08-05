package com.eignex.koblas

import com.eignex.koblas.dense.Blas
import com.eignex.koblas.dense.Lapack
import com.eignex.koblas.dense.ReferenceLinearAlgebra
import com.eignex.koblas.dense.VectorKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Detecting a silent fallback.
 *
 * koblas resolves backends without complaining: no host library means the portable kernels answer correctly
 * and nothing says so. That is the right default and a bad thing to discover in production, so these tests
 * pin the mechanism that makes it visible.
 *
 * Written against a registry cleared by `withCleanBackends`, because the whole subject is *what resolved* —
 * asserting anything here against whatever the platform happened to install would test the machine rather
 * than the library.
 */
class AccelerationTest {

    private class FakeHost(override val name: String) :
        Blas by ReferenceLinearAlgebra,
        Lapack by ReferenceLinearAlgebra {
        override val priority: Int get() = 100
    }

    /** Plausible host kernels: the arithmetic is irrelevant, being registered is the point. */
    private class FakeKernels(override val name: String = "fakeblas") : VectorKernels {
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
    fun `an empty registry reports every slot as portable`() = withCleanBackends {
        assertEquals(BackendSlot.entries.toSet(), koblas.portableSlots)
        for (slot in BackendSlot.entries) {
            assertTrue(!koblas.isAccelerated(slot), "$slot claimed acceleration with nothing registered")
        }
    }

    @Test
    fun `registering a host backend accelerates exactly its own slots`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        assertTrue(koblas.isAccelerated(BackendSlot.Blas))
        assertTrue(koblas.isAccelerated(BackendSlot.Lapack))
        // The vector and sparse slots are untouched by a Blas/Lapack registration.
        assertEquals(
            setOf(
                BackendSlot.VectorKernels,
                BackendSlot.SparseVectorKernels,
                BackendSlot.SparseBlas,
                BackendSlot.SparseLapack,
            ),
            koblas.portableSlots,
        )
    }

    /**
     * The compiled-in SIMD kernels are *portable*, however fast they are.
     *
     * This is the distinction that makes the check useful: `mathBackend` reporting `simd(8 lanes)` is not
     * evidence that a host library was found, and a caller who needs the host BLAS wants to know the
     * difference. Accelerated here means a registered backend is being consulted for long runs.
     */
    @Test
    fun `the compiled-in kernels do not count as acceleration`() = withCleanBackends {
        assertTrue(!koblas.isAccelerated(BackendSlot.VectorKernels), "the platform kernels are portable koblas")
        registerBackend(FakeKernels())
        assertTrue(koblas.isAccelerated(BackendSlot.VectorKernels), "a registered host half is acceleration")
    }

    @Test
    fun `requireAccelerated passes for the slots that are covered`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        koblas.requireAccelerated(BackendSlot.Blas, BackendSlot.Lapack)
        // Naming nothing checks nothing, which is what varargs mean here.
        koblas.requireAccelerated()
    }

    @Test
    fun `requireAccelerated names the slots that fell back and what filled them`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val failure = assertFailsWith<IllegalStateException> {
            koblas.requireAccelerated(BackendSlot.Blas, BackendSlot.SparseLapack)
        }
        val message = failure.message!!
        assertTrue("SparseLapack=reference" in message, "should name the slot and its backend: $message")
        assertTrue("Blas=" !in message, "should not name a slot that is accelerated: $message")
        assertTrue("backend=" in message, "should include the resolved summary: $message")
    }

    /**
     * A custom context is answered on its own terms, not the registry's.
     *
     * The point of #89 was that backend choice can be a value rather than process state, so the check has to
     * follow the value.
     */
    @Test
    fun `a context reports its own halves rather than the global registry`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val portable = koblas.with(blas = ReferenceLinearAlgebra, lapack = ReferenceLinearAlgebra)
        assertTrue(!portable.isAccelerated(BackendSlot.Blas), "the context's own half is the portable one")
        assertTrue(koblas.isAccelerated(BackendSlot.Blas), "the registry is still accelerated")
        assertFailsWith<IllegalStateException> { portable.requireAccelerated(BackendSlot.Blas) }
    }

    @Test
    fun `backendFor returns the half filling each slot`() = withCleanBackends {
        val host = FakeHost("openblas")
        registerBackend(host)
        assertSame(host, koblas.backendFor(BackendSlot.Blas))
        assertSame(host, koblas.backendFor(BackendSlot.Lapack))
        assertSame(koblas.vectorKernels, koblas.backendFor(BackendSlot.VectorKernels))
        assertSame(koblas.sparseBlas, koblas.backendFor(BackendSlot.SparseBlas))
    }
}
