package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.F64BuiltinKernels
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Guards the arms against resolving to an implementation other than the one they name.
 *
 * These run on whatever the machine has. A host library is present on some and not others, so the arms that
 * need one are attempted and skipped when the install reports it missing, which is the same signal a
 * benchmark run would get.
 */
@OptIn(ExperimentalKoblasApi::class)
class BenchmarkArmResolutionTest {
    private fun installedOrSkipped(install: () -> Unit): Boolean = try {
        install()
        true
    } catch (_: IllegalStateException) {
        false
    }

    @Test
    fun `each pinned kernel arm resolves to the provider it names`() {
        val pinned = buildList {
            add(SCALAR_KERNELS to F64BuiltinKernels.scalar)
            add(C_KERNELS to F64BuiltinKernels.c)
            add(SIMD_KERNELS to F64BuiltinKernels.simd)
        }
        for ((arm, provider) in pinned) {
            if (provider == null) continue
            installKernelProvider(arm)
            assertTrue(
                koblas.kernels.name.startsWith(arm),
                "the $arm arm resolved kernels to ${koblas.kernels.name}",
            )
        }
    }

    @Test
    fun `a pinned kernel arm does not inherit a discovered host half`() {
        val provider = F64BuiltinKernels.simd ?: F64BuiltinKernels.c ?: return
        val arm = if (provider === F64BuiltinKernels.simd) SIMD_KERNELS else C_KERNELS
        // Discovery first, which is what a benchmark process does before any arm asks for a pinned
        // provider, and what put a host half underneath the pinned arms.
        installDenseBackend(AUTOMATIC_BACKEND)
        installKernelProvider(arm)
        assertTrue(
            '+' !in koblas.kernels.name,
            "the $arm arm resolved kernels to ${koblas.kernels.name}, which joins a host half",
        )
        assertEquals(
            REFERENCE_BACKEND,
            koblas.blas.name,
            "the $arm arm left a non-portable matrix half installed, so a level-2 routine would not measure it",
        )
    }

    @Test
    fun `the reference dense arm is portable in every half it reaches`() {
        installDenseBackend(REFERENCE_BACKEND)
        assertEquals(REFERENCE_BACKEND, koblas.blas.name)
        assertEquals(REFERENCE_BACKEND, koblas.decompositions.name)
        assertTrue(
            '+' !in koblas.kernels.name,
            "the reference arm resolved kernels to ${koblas.kernels.name}, which routes to a host half",
        )
    }

    @Test
    fun `the host arms resolve to the host binding when one is installed`() {
        if (!installedOrSkipped { installDenseBackend(HOST_BACKEND) }) return
        assertEquals(hostBackendName, koblas.blas.name)
        installKernelProvider(HOST_BACKEND)
        assertEquals(hostBackendName, koblas.kernels.name)
    }

    @Test
    fun `an unknown arm is rejected rather than quietly measuring the installed one`() {
        assertFailsWith<IllegalStateException> { installKernelProvider("vectorised") }
        assertFailsWith<IllegalStateException> { installDenseBackend("fastest") }
    }
}
