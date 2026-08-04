package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.mathBackend
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
        resetRegisteredLinearAlgebra()
        try {
            block()
        } finally {
            resetRegisteredLinearAlgebra()
            incumbent?.let { registerLinearAlgebra(it) }
        }
    }

    @Test
    fun `an empty registry resolves to the reference backend`() {
        // There is one discovery path now: whatever a platform provides arrives through registration, so an
        // empty registry means the reference exactly, not "the reference or whatever discovery returned".
        withCleanRegistry {
            assertEquals(ReferenceLinearAlgebra, koblas)
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
        withCleanRegistry { assertTrue(koblas === ReferenceLinearAlgebra) }
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

    /**
     * The reset hook clears the install override as well as the registration. It did not always: the dense
     * one left an install in place while the sparse one cleared it, so the same call meant two different
     * things depending on which seam you were testing. `VectorKernelsTest` pins the level-1 half.
     */
    @Test
    fun `the reset hook clears the install override too`() {
        withCleanRegistry {
            installLinearAlgebra(Fake("manual", -1))
            resetRegisteredLinearAlgebra()
            assertTrue(koblas === ReferenceLinearAlgebra, "reset must clear the override, not just registration")
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
