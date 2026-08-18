package com.eignex.koblas

import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64Lapack
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.F64VectorKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

class AccelerationTest {

    private class FakeHost(override val name: String) :
        F64Blas by F64ReferenceLinearAlgebra,
        F64Lapack by F64ReferenceLinearAlgebra {
        override val priority: Int get() = 100
        override val isPortable: Boolean get() = false
        override val isAvailable: Boolean get() = true
        override val vectorKernels: F64VectorKernels get() = F64ReferenceLinearAlgebra.vectorKernels
    }

    private class FakeKernels(override val name: String = "fakeblas") : F64VectorKernels {
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
        assertTrue(koblas.isAccelerated(BackendSlot.F64Blas))
        assertTrue(koblas.isAccelerated(BackendSlot.F64Lapack))
        assertEquals(
            setOf(
                BackendSlot.F64VectorKernels,
                BackendSlot.F64SparseVectorKernels,
                BackendSlot.F64SparseBlas,
                BackendSlot.F64SparseLapack,
            ),
            koblas.portableSlots,
        )
    }

    @Test
    fun `the compiled-in kernels do not count as acceleration`() = withCleanBackends {
        assertTrue(!koblas.isAccelerated(BackendSlot.F64VectorKernels), "the platform kernels are portable koblas")
        registerBackend(FakeKernels())
        assertTrue(koblas.isAccelerated(BackendSlot.F64VectorKernels), "a registered host half is acceleration")
    }

    @Test
    fun `requireAccelerated passes for the slots that are covered`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        koblas.requireAccelerated(BackendSlot.F64Blas, BackendSlot.F64Lapack)
        koblas.requireAccelerated()
    }

    @Test
    fun `requireAccelerated names the slots that fell back and what filled them`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val failure = assertFailsWith<IllegalStateException> {
            koblas.requireAccelerated(BackendSlot.F64Blas, BackendSlot.F64SparseLapack)
        }
        val message = failure.message!!
        assertTrue("F64SparseLapack=reference" in message, "should name the slot and its backend: $message")
        assertTrue("F64Blas=" !in message, "should not name a slot that is accelerated: $message")
        assertTrue("backend=" in message, "should include the resolved summary: $message")
    }

    @Test
    fun `a context reports its own halves rather than the global registry`() = withCleanBackends {
        registerBackend(FakeHost("openblas"))
        val portable = koblas.with(blas = F64ReferenceLinearAlgebra, lapack = F64ReferenceLinearAlgebra)
        assertTrue(!portable.isAccelerated(BackendSlot.F64Blas), "the context's own half is the portable one")
        assertTrue(koblas.isAccelerated(BackendSlot.F64Blas), "the registry is still accelerated")
        assertFailsWith<IllegalStateException> { portable.requireAccelerated(BackendSlot.F64Blas) }
    }

    @Test
    fun `backendFor returns the half filling each slot`() = withCleanBackends {
        val host = FakeHost("openblas")
        registerBackend(host)
        assertSame(host, koblas.backendFor(BackendSlot.F64Blas))
        assertSame(host, koblas.backendFor(BackendSlot.F64Lapack))
        assertSame(koblas.vectorKernels, koblas.backendFor(BackendSlot.F64VectorKernels))
        assertSame(koblas.sparseBlas, koblas.backendFor(BackendSlot.F64SparseBlas))
    }
}
