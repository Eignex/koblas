package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Backend resolution: what [koblas] points at, and how registration, priority and an explicit install
 * interact.
 *
 * The registry is global process state, and on the targets where koblas ships a host-BLAS backend it is
 * already populated before any test runs. So these tests clear it and put the incumbent back rather than
 * assuming it is empty — without that, clearing it here would break every later test that expects the
 * platform's backend to be active.
 */
class BackendSelectionTest {

    private class Fake(override val name: String, override val priority: Int) : LinearAlgebra by ReferenceLinearAlgebra

    private class FakeBlas(override val name: String, override val priority: Int) : Blas by ReferenceLinearAlgebra

    private class FakeLapack(override val name: String, override val priority: Int) : Lapack by ReferenceLinearAlgebra

    /**
     * Runs [block] against an empty registry and no install, then restores whatever backend was resolved
     * before. Anything other than the reference must have been registered, since no test leaves an
     * install behind.
     */
    private fun withCleanRegistry(block: () -> Unit) {
        val incumbent = koblas.takeIf { it !== ReferenceLinearAlgebra }
        installLinearAlgebra(null)
        resetRegisteredLinearAlgebra()
        try {
            block()
        } finally {
            installLinearAlgebra(null)
            resetRegisteredLinearAlgebra()
            incumbent?.let { registerLinearAlgebra(it) }
        }
    }

    @Test
    fun `an empty registry resolves to the platform backend or the reference one`() {
        withCleanRegistry {
            assertEquals(platformLinearAlgebra() ?: ReferenceLinearAlgebra, koblas)
        }
    }

    @Test
    fun `registration keeps the highest priority and install overrides everything`() {
        withCleanRegistry {
            registerLinearAlgebra(Fake("low", 5))
            assertEquals("low", koblas.name)
            registerLinearAlgebra(Fake("high", 50))
            assertEquals("high", koblas.name)
            registerLinearAlgebra(Fake("mid", 20)) // weaker than the incumbent: ignored
            assertEquals("high", koblas.name)
            installLinearAlgebra(Fake("manual", -1)) // an explicit install beats any priority
            assertEquals("manual", koblas.name)
            installLinearAlgebra(null)
            assertEquals("high", koblas.name)
        }
    }

    @Test
    fun `the incumbent backend survives a cleared registry`() {
        // The restore path above is what keeps the host-BLAS suites valid whatever order tests run in.
        val before = koblas
        withCleanRegistry { assertTrue(koblas === platformLinearAlgebra() || koblas === ReferenceLinearAlgebra) }
        assertEquals(before, koblas)
    }

    @Test
    fun `the halves are selected independently and composed`() {
        withCleanRegistry {
            registerBlas(FakeBlas("fastblas", 50))
            assertEquals("fastblas+reference", koblas.name)
            // The LAPACK half still answers, from the portable implementation.
            assertEquals(2, koblas.factor(DenseMatrix.diagonal(2)).n)
            registerLapack(FakeLapack("fastlapack", 10))
            assertEquals("fastblas+fastlapack", koblas.name)
            // Each half keeps its own ranking: a weaker BLAS does not displace the stronger one.
            registerBlas(FakeBlas("slowblas", 5))
            assertEquals("fastblas+fastlapack", koblas.name)
        }
    }

    @Test
    fun `a backend providing both halves is used directly`() {
        withCleanRegistry {
            val both = Fake("whole", 30)
            registerLinearAlgebra(both)
            // Not wrapped in a composite: one object won both, so the name is its own and calls reach it
            // without a second hop.
            assertSame(both, koblas)
            assertEquals("whole", koblas.name)
        }
    }

    @Test
    fun `koblasInfo reports both seams`() {
        assertEquals("backend=${koblas.name}, primitives=$mathBackend", koblasInfo)
    }

    @Test
    fun `mathBackend identifies the primitive kernels`() {
        // The other seam: which level-1 kernels the platform resolved, independent of the backend.
        assertTrue(mathBackend.isNotEmpty())
    }
}
