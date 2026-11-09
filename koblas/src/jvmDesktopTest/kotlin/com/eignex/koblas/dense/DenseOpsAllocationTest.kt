package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.operations.dotVector
import com.eignex.koblas.testutil.allocation.*
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/**
 * Default-engine probes bound view adaptation around accelerated calls. Portable correctness and scratch
 * reuse belong to conformance tests. Uninstrumented SIMD tasks measure numerical kernels and complete
 * accelerated calls without coverage instrumentation changing carrier allocation.
 */
@RunWith(Parameterized::class)
class DenseOpsAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(
            levelTwoCases() + workspaceCases() + unlentCases(),
        )

        private fun levelTwoCases(): List<AllocationCase> = listOf(
            with(StridedDotFixture()) {
                AllocationCase("strided dense dot allocates nothing", 1_000) { engine.dotVector(x, y) }
            },
            with(ContiguousFixture()) {
                AllocationCase("contiguous gemv allocates nothing", 1_000) {
                    koblas.gemv(
                        1e-12,
                        a,
                        contiguous,
                        1.0,
                        destination,
                    )
                    destination
                }
            },
            with(ContiguousFixture()) {
                AllocationCase("contiguous symv allocates nothing", 1_000) {
                    koblas.symv(
                        1e-12,
                        a,
                        contiguous,
                        1.0,
                        destination,
                    )
                    destination
                }
            },
        )

        private fun workspaceCases(): List<AllocationCase> = listOf(
            with(StridedOperandFixture()) {
                AllocationCase("a strided operand lent a workspace allocates nothing", 1_000) {
                    koblas.gemv(
                        1e-12,
                        a,
                        strided,
                        1.0,
                        destination,
                        scratch = workspace,
                    )
                    destination
                }
            },
        )

        // Without a workspace a strided operand may need one gather. A portable default may need none.
        private fun unlentCases(): List<AllocationCase> = listOf(
            with(StridedOperandFixture()) {
                val oneGather = n * Double.SIZE_BYTES + ARRAY_HEADER_BYTES
                AllocationCase("a strided operand is gathered once per call", 1_000, oneGather + FLOOR_BYTES) {
                    koblas.gemv(
                        1e-12,
                        a,
                        strided,
                        1.0,
                        destination,
                    )
                    destination
                }
            },

        )
    }

    private class StridedDotFixture {
        val engine = KoblasEngine.simd ?: KoblasEngine.scalar
        val n = 256
        val x = DenseVector.wrap(DoubleArray(2 * n) { it * 0.01 }, 0, n, 2)
        val y = DenseVector.wrap(DoubleArray(2 * n) { 1.0 / (it + 1) }, 0, n, 2)
    }

    private class ContiguousFixture {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val contiguous = DenseVector.wrap(DoubleArray(n) { 1.0 + (it % 7) * 0.25 })
        val destination = DenseVector.wrap(DoubleArray(n))
    }

    /** A gemv whose input vector is a strided view; the workspace is warmed with one call. */
    private class StridedOperandFixture {
        val n = 128
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val strided = DenseVector.wrap(DoubleArray(2 * n) { 1.0 + (it % 7) * 0.25 }, 0, n, 2)
        val destination = DenseVector.wrap(DoubleArray(n))
        val workspace = Scratch()
        init {
            koblas.gemv(
                1e-12,
                a,
                strided,
                1.0,
                destination,
                scratch = workspace,
            )
        }
    }
}
