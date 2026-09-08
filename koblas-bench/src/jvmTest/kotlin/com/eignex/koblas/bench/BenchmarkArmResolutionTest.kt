package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.F64BuiltinKernels
import com.eignex.koblas.core.F64DenseMatrix
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
    }

    @Test
    fun `the host arm resolves to the host binding when one is installed`() {
        if (!installedOrSkipped { installDenseBackend(HOST_BACKEND) }) return
        assertEquals(hostBackendName, koblas.blas.name)
    }

    @Test
    fun `the built in dense arm bypasses discovery`() {
        val arm = DenseBenchmarkArm.resolve(BUILTIN_BACKEND)
        assertTrue(arm.context != null)
        assertEquals(null, arm.external)
        assertTrue(arm.identity.startsWith("built-in/reference/"), arm.identity)
    }

    @Test
    fun `the openblas arm is benchmark owned and single threaded`() {
        val comparator = openBlasComparator() ?: return
        assertTrue(comparator.identity.startsWith(OPENBLAS_BACKEND))
        assertEquals("1 thread", comparator.threading)
        assertTrue(comparator !== koblas.blas)
    }

    @Test
    fun `the onemkl dense arm is benchmark owned and single threaded when available`() {
        val comparator = oneMklDenseComparator() ?: return
        assertTrue(comparator.identity.startsWith(ONEMKL_BACKEND))
        assertEquals("1 thread", comparator.threading)
        assertTrue(comparator !== koblas.blas)
    }

    @Test
    fun `external dense level one agrees with built in`() {
        val external = openBlasComparator() ?: return
        val context = explicitBuiltInContext()
        val x = doubleArrayOf(0.25, -2.0, 3.5, 0.0, 8.0)
        val y = doubleArrayOf(-4.0, 1.5, 2.0, -7.0, 0.125)
        assertEquals(context.kernels.dot(x, 0, y, 0, x.size), external.dot(x, y), 1e-12)
        assertEquals(context.kernels.nrm2(x, 0, x.size), external.nrm2(x), 1e-12)
        assertEquals(context.kernels.asum(x, 0, x.size), external.asum(x), 1e-12)
    }

    @Test
    fun `external dense matrix calls agree with built in`() {
        val external = openBlasComparator() ?: return
        val context = explicitBuiltInContext()
        val rng = benchRng()
        val a = randomMatrix(7, 5, rng)
        val b = randomMatrix(5, 3, rng)
        val expected = F64DenseMatrix.zero(7, 3)
        val actual = F64DenseMatrix.zero(7, 3)
        context.gemm(1.25, a, false, b, false, 0.0, expected)
        external.gemm(1.25, a, false, b, false, 0.0, actual)
        for (i in actual.data.indices) assertEquals(expected.data[i], actual.data[i], 1e-11, "entry $i")
    }

    @Test
    fun `onemkl sparse prepared matrix agrees with built in when available`() {
        val external = oneMklSparseComparator() ?: return
        val context = explicitBuiltInContext()
        val a = sparseComparisonMatrix(17, 0.15, "skewed", benchRng())
        val x = randomVector(a.cols, benchRng())
        val expected = DoubleArray(a.rows)
        val actual = DoubleArray(a.rows)
        context.sparseBlas.gemv(1.0, a, x, 0.0, expected)
        external.prepare(a).use { it.gemv(1.0, x, 0.0, actual) }
        for (i in actual.indices) assertEquals(expected[i], actual[i], 1e-11, "entry $i")
    }

    @Test
    fun `an unknown arm is rejected rather than quietly measuring the installed one`() {
        assertFailsWith<IllegalStateException> { installKernelProvider("vectorised") }
        assertFailsWith<IllegalStateException> { installDenseBackend("fastest") }
    }
}
