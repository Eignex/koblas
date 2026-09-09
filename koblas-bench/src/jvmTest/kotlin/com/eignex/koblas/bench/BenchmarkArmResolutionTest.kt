package com.eignex.koblas.bench

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.BuiltinKernels
import com.eignex.koblas.DenseMatrix
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
    @Test
    fun `each pinned kernel arm resolves to the provider it names`() {
        val pinned = buildList {
            add(SCALAR_KERNELS to BuiltinKernels.scalar)
            add(C_KERNELS to BuiltinKernels.c)
            add(SIMD_KERNELS to BuiltinKernels.simd)
        }
        for ((arm, provider) in pinned) {
            if (provider == null) continue
            val context = kernelEngine(arm)
            assertTrue(
                context.kernels.name.startsWith(arm),
                "the $arm arm resolved kernels to ${context.kernels.name}",
            )
        }
    }

    @Test
    fun `a pinned kernel arm does not inherit a discovered host half`() {
        val provider = BuiltinKernels.simd ?: BuiltinKernels.c ?: return
        val arm = if (provider === BuiltinKernels.simd) SIMD_KERNELS else C_KERNELS
        // Discovery first, which is what a benchmark process does before any arm asks for a pinned
        // provider, and what put a host half underneath the pinned arms.
        kernelEngine(AUTOMATIC_KERNELS)
        val context = kernelEngine(arm)
        assertTrue(
            '+' !in context.kernels.name,
            "the $arm arm resolved kernels to ${context.kernels.name}, which joins a host half",
        )
        assertEquals(
            BUILTIN_BACKEND,
            context.blas.name,
            "the $arm arm left a non-portable matrix half installed, so a level-2 routine would not measure it",
        )
    }

    @Test
    fun `the built in dense arm bypasses discovery`() {
        val arm = DenseBenchmarkArm.resolve(BUILTIN_BACKEND)
        assertTrue(arm.context != null)
        assertEquals(null, arm.external)
        assertTrue(arm.identity.startsWith("built-in/built-in/"), arm.identity)
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
        val expected = DenseMatrix.zero(7, 3)
        val actual = DenseMatrix.zero(7, 3)
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
    fun `onemkl triangular benchmark rows agree with built in across repeated calls`() {
        if (oneMklSparseComparator() == null) return
        val builtIn = sparseProductBenchmark(BUILTIN_BACKEND)
        val oneMkl = sparseProductBenchmark(ONEMKL_BACKEND)
        try {
            repeat(2) { invocation ->
                assertVectorNear(builtIn.trsv(), oneMkl.trsv(), "trsv invocation $invocation")
                assertVectorNear(builtIn.trmv(), oneMkl.trmv(), "trmv invocation $invocation")
                assertMatrixNear(builtIn.trsm(), oneMkl.trsm(), "trsm invocation $invocation")
                assertMatrixNear(builtIn.trmm(), oneMkl.trmm(), "trmm invocation $invocation")
                assertMatrixNear(builtIn.trmmRight(), oneMkl.trmmRight(), "trmm right invocation $invocation")
            }
        } finally {
            builtIn.tearDown()
            oneMkl.tearDown()
        }
    }

    @Test
    fun `onemkl prepared sparse product agrees with built in across repeated calls`() {
        if (oneMklSparseComparator() == null) return
        val builtIn = sparseProductBenchmark(BUILTIN_BACKEND)
        val oneMkl = sparseProductBenchmark(ONEMKL_BACKEND)
        try {
            repeat(2) { invocation ->
                val expected = builtIn.preparedSparseProduct()
                val actual = oneMkl.preparedSparseProduct()
                assertEquals(expected.rows, actual.rows, "rows invocation $invocation")
                assertEquals(expected.cols, actual.cols, "cols invocation $invocation")
                for (j in 0 until expected.cols) for (i in 0 until expected.rows) {
                    assertEquals(expected[i, j], actual[i, j], 1e-10, "entry ($i, $j) invocation $invocation")
                }
            }
        } finally {
            builtIn.tearDown()
            oneMkl.tearDown()
        }
    }

    @Test
    fun `an unknown arm is rejected rather than quietly measuring the installed one`() {
        assertFailsWith<IllegalStateException> { kernelEngine("vectorised") }
    }

    private fun sparseProductBenchmark(arm: String): SparseProductHostBenchmark = SparseProductHostBenchmark().also {
        it.n = 31
        it.sparseArm = arm
        it.density = 0.1
        it.productShape = "regular"
        it.setup()
    }

    private fun assertVectorNear(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertEquals(expected.size, actual.size, "$context size")
        for (i in expected.indices) assertEquals(expected[i], actual[i], 1e-10, "$context entry $i")
    }

    private fun assertMatrixNear(expected: DenseMatrix, actual: DenseMatrix, context: String) {
        assertEquals(expected.rows, actual.rows, "$context rows")
        assertEquals(expected.cols, actual.cols, "$context cols")
        for (i in expected.data.indices) assertEquals(expected.data[i], actual.data[i], 1e-10, "$context entry $i")
    }
}
