package com.eignex.koblas.dense

import com.eignex.koblas.Backend
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.installBackends
import com.eignex.koblas.koblas
import com.eignex.koblas.koblasInfo
import com.eignex.koblas.mathBackend
import com.eignex.koblas.registerBackend
import com.eignex.koblas.resetBackends
import com.eignex.koblas.withCleanBackends
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/** Backend resolution: what [koblas] resolves to, and how registration, priority and an explicit install interact. */
class BackendSelectionTest {

    private class Fake(override val name: String, override val priority: Int) : LinearAlgebra by ReferenceLinearAlgebra

    private class FakeBlas(override val name: String, override val priority: Int) : Blas by ReferenceLinearAlgebra

    private class FakeLapack(override val name: String, override val priority: Int) : Lapack by ReferenceLinearAlgebra

    /** Implements no half at all, so registering it is a mistake worth reporting. */
    private class NotABackend(override val name: String = "nothing") : Backend

    @Test
    fun `an empty registry resolves to the reference backend`() {
        // There is one discovery path: whatever a platform provides arrives through registration, so an
        // empty registry means the reference exactly, not "the reference or whatever discovery returned".
        withCleanBackends {
            assertSame(ReferenceLinearAlgebra, koblas.blas)
            assertSame(ReferenceLinearAlgebra, koblas.lapack)
            assertEquals("reference", koblas.name)
        }
    }

    @Test
    fun `registration keeps the highest priority and install overrides everything`() {
        withCleanBackends {
            // The sparse halves stay on the reference, and the name says so -- a dense-only backend does
            // not silently claim the whole context.
            registerBackend(Fake("low", 5))
            assertEquals("low+reference", koblas.name)
            registerBackend(Fake("high", 50))
            assertEquals("high+reference", koblas.name)
            registerBackend(Fake("mid", 20)) // weaker than the incumbent: ignored
            assertEquals("high+reference", koblas.name)
            // An explicit install beats any priority. `with` starts from what resolved and replaces halves.
            val manual = Fake("manual", -1)
            installBackends(koblas.with(blas = manual, lapack = manual))
            assertEquals("manual+reference", koblas.name)
            installBackends(null)
            assertEquals("high+reference", koblas.name)
        }
    }

    @Test
    fun `the incumbent backend survives a cleared registry`() {
        // The restore path above is what keeps the host-BLAS suites valid whatever order tests run in.
        val before = koblas.blas
        withCleanBackends { assertSame(ReferenceLinearAlgebra, koblas.blas) }
        assertSame(before, koblas.blas)
    }

    @Test
    fun `the halves are selected independently`() {
        withCleanBackends {
            registerBackend(FakeBlas("fastblas", 50))
            assertEquals("fastblas+reference", koblas.name)
            // The LAPACK half still answers, from the portable implementation.
            assertEquals(2, koblas.factor(DenseMatrix.diagonal(2)).n)
            registerBackend(FakeLapack("fastlapack", 10))
            assertEquals("fastblas+fastlapack+reference", koblas.name)
            // Each half keeps its own ranking: a weaker BLAS does not displace the stronger one.
            registerBackend(FakeBlas("slowblas", 5))
            assertEquals("fastblas+fastlapack+reference", koblas.name)
        }
    }

    /** One object registered for several halves lands in each of them. */
    @Test
    fun `a backend providing both halves is used for both`() {
        withCleanBackends {
            val both = Fake("whole", 30)
            registerBackend(both)
            assertSame(both, koblas.blas)
            assertSame(both, koblas.lapack)
            assertEquals("whole+reference", koblas.name, "the dense halves are its; the sparse ones are not")
        }
    }

    /** Registering something that implements no half is a mistake, not a silent no-op. */
    @Test
    fun `registering a backend that implements no half fails loudly`() {
        withCleanBackends {
            val failure = assertFailsWith<IllegalArgumentException> { registerBackend(NotABackend()) }
            assertTrue(failure.message!!.contains("nothing"), "the message should name the backend")
        }
    }

    /** The reset hook clears the install override as well as the registration. */
    @Test
    fun `the reset hook clears the install override too`() {
        withCleanBackends {
            val manual = Fake("manual", -1)
            installBackends(koblas.with(blas = manual, lapack = manual))
            resetBackends()
            assertSame(ReferenceLinearAlgebra, koblas.blas, "reset must clear the override, not just registration")
        }
    }

    @Test
    fun `koblasInfo reports the backend and the kernels`() {
        assertEquals("backend=${koblas.name}, kernels=${koblas.vectorKernels.name}", koblasInfo)
    }

    @Test
    fun `mathBackend identifies the vector kernels`() {
        assertTrue(mathBackend.isNotEmpty())
        assertEquals(koblas.vectorKernels.name, mathBackend)
    }
}
