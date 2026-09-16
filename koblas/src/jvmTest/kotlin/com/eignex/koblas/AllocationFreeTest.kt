package com.eignex.koblas

import com.eignex.koblas.testutil.allocation.bytesPerIteration
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The seams a caller may run inside a hot loop without the collector noticing.
 *
 * Level 1 and the sparse primitives take raw arrays and offsets precisely so a caller sweeping a matrix can
 * reach them per sub-range without wrapping anything, and this is what holds them to it. Level 2 and 3 make no
 * such promise now that they are whole vendor calls: each one confines an arena for the duration of the call,
 * which is the price of the boundary and is paid once per call rather than once per element.
 */
class AllocationFreeTest {

    private companion object {
        /** Allowance for effects that are not koblas's (instrumentation, index boxing, JIT noise). */
        const val FLOOR_BYTES = 64.0

        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
    }

    @Test
    fun `dense level one kernels allocate nothing`() {
        val n = 512
        val a = DoubleArray(n) { it * 0.01 }
        val b = DoubleArray(n) { 1.0 / (it + 1) }
        val kernels = engine.vectorKernels

        val dot = bytesPerIteration(1_000) { kernels.dot(a, 0, b, 0, n) }
        val axpy = bytesPerIteration(1_000) { kernels.axpy(b, 0, 0.5, a, 0, n) }
        val nrm2 = bytesPerIteration(1_000) { kernels.nrm2(a, 0, n) }
        val strided = bytesPerIteration(1_000) { kernels.dot(a, 0, b, 0, n / 2, 2, 2) }

        assertTrue(dot <= FLOOR_BYTES, "dot allocated $dot B per call")
        assertTrue(axpy <= FLOOR_BYTES, "axpy allocated $axpy B per call")
        assertTrue(nrm2 <= FLOOR_BYTES, "nrm2 allocated $nrm2 B per call")
        assertTrue(strided <= FLOOR_BYTES, "a strided dot allocated $strided B per call")
    }

    @Test
    fun `sparse vector kernels allocate nothing`() {
        val n = 128
        val vector = SparseVector.of(n, IntArray(n / 2) { it * 2 }, DoubleArray(n / 2) { it + 1.0 })
        val x = DoubleArray(n) { it * 0.01 }

        val levelOneBytes = bytesPerIteration(1_000) { engine.sparseKernels.dot(vector, x) }

        assertTrue(levelOneBytes <= FLOOR_BYTES, "sparse level one allocated $levelOneBytes B per call")
    }
}
