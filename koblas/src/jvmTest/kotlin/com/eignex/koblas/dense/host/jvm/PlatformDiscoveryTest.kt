package com.eignex.koblas.dense.host.jvm

import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.dense.F64Blas
import com.eignex.koblas.dense.F64ReferenceLinearAlgebra
import com.eignex.koblas.dense.host.jvm.HostBlasCalls
import com.eignex.koblas.internal.backend.BackendSlot
import com.eignex.koblas.internal.backend.f64DispatchThresholds
import com.eignex.koblas.internal.backend.probe
import com.eignex.koblas.internal.backend.registerPlatformBackends
import com.eignex.koblas.isAccelerated
import com.eignex.koblas.koblas
import com.eignex.koblas.portableSlots
import com.eignex.koblas.testutil.host.HostLibraryTest
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
                koblas.isAccelerated(BackendSlot.F64Blas),
                "the F64Blas slot should be accelerated exactly when a host CBLAS resolved",
            )
            assertEquals(
                HostBlasCalls.lapackAvailable,
                koblas.isAccelerated(BackendSlot.F64Lapack),
                "the F64Lapack slot should be accelerated exactly when a host LAPACKE resolved",
            )
            // The vector kernels have no host implementation to route to, so they stay portable either way.
            assertTrue(
                BackendSlot.F64VectorKernels in koblas.portableSlots,
                "koblas ships no host vector kernels, so that slot cannot be accelerated",
            )
        }
    }

    /**
     * The probe is what keeps a candidate whose native library is absent or broken out of the registry. It
     * has to reject a wrong answer as firmly as a throw, since a backend that loads and computes nonsense is
     * the worse of the two, and it has to reach the candidate's native path to see either.
     *
     * Each fake overrides the primitive rather than a convenience overload, because that is what the probe
     * calls and because a backend assembled by delegation inherits a forwarder for everything it leaves
     * alone.
     */
    @Test
    fun `the probe accepts a working backend and rejects every broken one`() {
        assertTrue(probe(F64ReferenceLinearAlgebra), "the reference backend should pass its own probe")

        assertTrue(
            !probe(GemmBackend { _, _, c -> c.data.fill(0.0) }),
            "a backend computing the wrong product should be rejected",
        )
        assertTrue(
            !probe(GemmBackend { _, _, _ -> throw UnsatisfiedLinkError("no native library") }),
            "a backend whose native library is missing should be rejected",
        )
        // Shaped like koblas's own adapters: portable below its gate, native at or above it. A probe that
        // stays under every gate never reaches the half that can be broken.
        assertTrue(
            !probe(
                GemmBackend { a, reference, c ->
                    if (a.rows >= f64DispatchThresholds.level3) throw UnsatisfiedLinkError("no native library")
                    reference()
                    @Suppress("UNUSED_EXPRESSION")
                    c
                },
            ),
            "a backend whose native path is missing should be rejected",
        )
    }

    /** A backend whose gemm is [gemm]; every other routine is the reference's. */
    private class GemmBackend(
        private val gemm: (a: F64DenseMatrix, reference: () -> Unit, c: F64DenseMatrix) -> Unit,
    ) : F64Blas by F64ReferenceLinearAlgebra {
        override val name: String get() = "fake"

        @Suppress("LongParameterList") // the BLAS dgemm signature
        override fun gemm(
            alpha: Double,
            a: F64DenseMatrix,
            transposeA: Boolean,
            b: F64DenseMatrix,
            transposeB: Boolean,
            beta: Double,
            c: F64DenseMatrix,
        ) = gemm(
            a,
            { F64ReferenceLinearAlgebra.gemm(alpha, a, transposeA, b, transposeB, beta, c) },
            c,
        )
    }

    private companion object {
        const val PROPERTY = "koblas.backend"
    }
}
