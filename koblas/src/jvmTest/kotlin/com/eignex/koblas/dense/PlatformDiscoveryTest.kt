package com.eignex.koblas.dense

import com.eignex.koblas.BackendSlot
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.HostLibraryTest
import com.eignex.koblas.hostblas.HostBlasCalls
import com.eignex.koblas.isAccelerated
import com.eignex.koblas.koblas
import com.eignex.koblas.portableSlots
import com.eignex.koblas.withCleanBackends
import org.junit.experimental.categories.Category
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Discovery itself, which the everyday run never reaches: it pins `koblas.backend=reference`, whose first
 * line returns before anything is registered. Carries the host-library category because every path past that
 * line consults [HostBlasCalls], and loading a host library is what the default run is kept away from.
 */
@Category(HostLibraryTest::class)
class PlatformDiscoveryTest {

    /** Runs [block] with `koblas.backend` set to [value], or absent when null, then puts it back. */
    private fun withRequestedBackend(value: String?, block: () -> Unit) {
        val previous: String? = System.getProperty(PROPERTY)
        if (value == null) System.clearProperty(PROPERTY) else System.setProperty(PROPERTY, value)
        try {
            block()
        } finally {
            if (previous == null) System.clearProperty(PROPERTY) else System.setProperty(PROPERTY, previous)
        }
    }

    @Test
    fun `asking for the reference backend registers nothing at all`() = withCleanBackends {
        withRequestedBackend("reference") {
            registerPlatformBackends()
            assertEquals(
                BackendSlot.entries.toSet(),
                koblas.portableSlots,
                "reference was requested, so every slot should still be portable",
            )
        }
    }

    /** A name nothing answers to leaves the registry empty, rather than falling through to a host backend. */
    @Test
    fun `asking for a backend that does not exist registers nothing`() = withCleanBackends {
        withRequestedBackend("no-such-backend") {
            registerPlatformBackends()
            assertEquals(
                BackendSlot.entries.toSet(),
                koblas.portableSlots,
                "an unmatched name should leave every slot portable",
            )
        }
    }

    /**
     * With nothing requested, discovery registers whatever the machine has. The assertion follows the
     * machine rather than asserting one outcome, since a box without the libraries is a valid one.
     */
    @Test
    fun `discovery with nothing requested registers what the machine provides`() = withCleanBackends {
        withRequestedBackend(null) {
            registerPlatformBackends()
            assertEquals(
                HostBlasCalls.available,
                koblas.isAccelerated(BackendSlot.Blas),
                "the Blas slot should be accelerated exactly when a host CBLAS resolved",
            )
            assertEquals(
                HostBlasCalls.lapackAvailable,
                koblas.isAccelerated(BackendSlot.Lapack),
                "the Lapack slot should be accelerated exactly when a host LAPACKE resolved",
            )
            // The vector kernels have no host implementation to route to, so they stay portable either way.
            assertTrue(
                BackendSlot.VectorKernels in koblas.portableSlots,
                "koblas ships no host vector kernels, so that slot cannot be accelerated",
            )
        }
    }

    /**
     * The probe is what keeps a candidate whose native library is absent or broken out of the registry. It
     * has to reject a wrong answer as firmly as a throw, since a backend that loads and computes nonsense is
     * the worse of the two.
     */
    @Test
    fun `the probe accepts a working backend and rejects every broken one`() {
        assertTrue(probe(ReferenceLinearAlgebra), "the reference backend should pass its own probe")

        val wrongAnswer = object : Blas by ReferenceLinearAlgebra {
            override fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean) = doubleArrayOf(0.0)
        }
        assertTrue(!probe(wrongAnswer), "a backend computing the wrong product should be rejected")

        val wrongLength = object : Blas by ReferenceLinearAlgebra {
            override fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean) = DoubleArray(0)
        }
        assertTrue(!probe(wrongLength), "a backend returning the wrong shape should be rejected")

        val throwing = object : Blas by ReferenceLinearAlgebra {
            override fun gemv(a: DenseMatrix, x: DoubleArray, transpose: Boolean): DoubleArray =
                throw UnsatisfiedLinkError("no native library")
        }
        assertTrue(!probe(throwing), "a backend whose native library is missing should be rejected")
    }

    private companion object {
        const val PROPERTY = "koblas.backend"
    }
}
