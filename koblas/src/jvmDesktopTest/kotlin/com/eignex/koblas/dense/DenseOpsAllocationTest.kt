package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.dotVector
import com.eignex.koblas.testutil.allocation.ARRAY_HEADER_BYTES
import com.eignex.koblas.testutil.allocation.AllocationCase
import com.eignex.koblas.testutil.allocation.FLOOR_BYTES
import com.eignex.koblas.testutil.allocation.parameterRows
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/**
 * Dense Level 2 and 3 operations allocate nothing once warmed, or only the one buffer a call without a
 * workspace has to take. Portable probes cover the shared scheduling; instrumentation prevents Vector API
 * carrier replacement, so the uninstrumented SIMD allocation tasks measure the vector panels separately.
 */
@RunWith(Parameterized::class)
class DenseOpsAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(
            levelTwoCases() + workspaceCases() + productCases() + unlentCases(),
        )

        private fun levelTwoCases(): List<AllocationCase> = listOf(
            with(StridedDotFixture()) {
                AllocationCase("strided dense dot allocates nothing", 1_000) { engine.dotVector(x, y) }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense gemv allocates nothing", 400) {
                    portable.gemv(1e-12, a, x, 0, 1, 1.0, y, 0, 1)
                    y
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("transposed dense gemv allocates nothing", 400) {
                    portable.gemv(
                        1e-12,
                        a,
                        x,
                        0,
                        1,
                        1.0,
                        y,
                        0,
                        1,
                        transpose = true,
                    )
                    y
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense symv allocates nothing", 400) {
                    portable.symv(1e-12, a, x, 0, 1, 1.0, y, 0, 1)
                    y
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense ger allocates nothing", 400) {
                    portable.ger(1e-12, x, 0, 1, y, 0, 1, a)
                    a
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense syr allocates nothing", 400) {
                    portable.syr(1e-12, view.values, view.offset, view.stride, a)
                    a
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense trmv allocates nothing", 400) {
                    portable.trmv(triangle, y, 0, 1, lower = true)
                    y
                }
            },
            with(DenseLevelTwoFixture()) {
                AllocationCase("dense trsv allocates nothing", 400) {
                    portable.trsv(triangle, y, 0, 1, lower = true)
                    y
                }
            },
            with(ContiguousFixture()) {
                AllocationCase("contiguous gemv allocates nothing", 1_000) {
                    koblas.gemv(
                        1e-12,
                        a,
                        contiguous.values,
                        contiguous.offset,
                        contiguous.stride,
                        1.0,
                        destination.values,
                        destination.offset,
                        destination.stride,
                    )
                    destination
                }
            },
            with(ContiguousFixture()) {
                AllocationCase("contiguous symv allocates nothing", 1_000) {
                    koblas.symv(
                        1e-12,
                        a,
                        contiguous.values,
                        contiguous.offset,
                        contiguous.stride,
                        1.0,
                        destination.values,
                        destination.offset,
                        destination.stride,
                    )
                    destination
                }
            },
        )

        private fun workspaceCases(): List<AllocationCase> = listOf(
            with(AliasedLevelTwoFixture()) {
                AllocationCase("aliased dense gemv reuses a workspace", 400) {
                    portable.gemv(
                        1e-12,
                        a,
                        y,
                        0,
                        1,
                        1.0,
                        y,
                        0,
                        1,
                        scratch = workspace,
                    )
                    y
                }
            },
            with(AliasedLevelTwoFixture()) {
                AllocationCase("aliased dense symv reuses a workspace", 400) {
                    portable.symv(
                        1e-12,
                        a,
                        y,
                        0,
                        1,
                        1.0,
                        y,
                        0,
                        1,
                        scratch = workspace,
                    )
                    y
                }
            },
            with(StridedOperandFixture()) {
                AllocationCase("a strided operand lent a workspace allocates nothing", 1_000) {
                    koblas.gemv(
                        1e-12,
                        a,
                        strided.values,
                        strided.offset,
                        strided.stride,
                        1.0,
                        destination.values,
                        destination.offset,
                        destination.stride,
                        scratch = workspace,
                    )
                    destination
                }
            },
            with(DenseLevelThreeFixture()) {
                AllocationCase("dense trsm reuses a workspace", 100) {
                    portable.trsm(triangle, b, lower = true, scratch = workspace)
                    b
                }
            },
            with(DenseLevelThreeFixture()) {
                AllocationCase("dense trmm reuses a workspace", 100) {
                    portable.trmm(triangle, b, lower = true, scratch = workspace)
                    b
                }
            },
        )

        private fun productCases(): List<AllocationCase> = listOf(
            with(DenseProductFixture()) {
                AllocationCase("shallow matrix products reuse a workspace", 50) {
                    portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, scratch = workspace)
                    packedTarget
                }
            },
            with(DenseProductFixture()) {
                AllocationCase("deep matrix products reuse a workspace", 50) {
                    portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, scratch = workspace)
                    deepTarget
                }
            },
            with(DenseProductFixture()) {
                AllocationCase("unpacked matrix products reuse a workspace", 200) {
                    portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, scratch = workspace)
                    smallTarget
                }
            },
        )

        // A call lent no workspace takes what it needs fresh: a strided operand is gathered once, and an
        // unpacked product accumulates one destination column before spending the multipliers, which is what
        // keeps alpha on a sum of products. The bounds also admit implementations that take nothing.
        private fun unlentCases(): List<AllocationCase> = listOf(
            with(StridedOperandFixture()) {
                val oneGather = n * Double.SIZE_BYTES + ARRAY_HEADER_BYTES
                AllocationCase("a strided operand is gathered once per call", 1_000, oneGather + FLOOR_BYTES) {
                    koblas.gemv(
                        1e-12,
                        a,
                        strided.values,
                        strided.offset,
                        strided.stride,
                        1.0,
                        destination.values,
                        destination.offset,
                        destination.stride,
                    )
                    destination
                }
            },
            with(UnpackedProductFixture()) {
                val oneColumn = order * Double.SIZE_BYTES + ARRAY_HEADER_BYTES
                AllocationCase("an unpacked product takes one destination column", 500, oneColumn + FLOOR_BYTES) {
                    portable.gemm(1e-12, a, false, b, false, 1.0, c)
                    c
                }
            },
        )
    }

    private class StridedDotFixture {
        val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar
        val n = 256
        val x = StridedVector(DoubleArray(2 * n) { it * 0.01 }, 0, n, 2)
        val y = StridedVector(DoubleArray(2 * n) { 1.0 / (it + 1) }, 0, n, 2)
    }

    private class DenseLevelTwoFixture {
        val n = 48
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val triangle = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 0.25 + (it % 11) * 0.0625 }).also {
            for (i in 0 until n) it.values[i + i * n] = 4.0
        }
        val x = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val y = DoubleArray(n) { 0.5 }
        val view = DenseVector.wrap(x)
        val portable = BuiltinEngines.scalar
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
        val strided = StridedVector(DoubleArray(2 * n) { 1.0 + (it % 7) * 0.25 }, 0, n, 2)
        val destination = DenseVector.wrap(DoubleArray(n))
        val workspace = Scratch()
        init {
            koblas.gemv(
                1e-12,
                a,
                strided.values,
                strided.offset,
                strided.stride,
                1.0,
                destination.values,
                destination.offset,
                destination.stride,
                scratch = workspace,
            )
        }
    }

    private class AliasedLevelTwoFixture {
        val n = 128
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(n, n, DoubleArray(n * n) { 1.0 + (it % 13) * 0.125 })
        val y = DoubleArray(n) { 1.0 + (it % 7) * 0.25 }
        val workspace = Scratch()
        init {
            portable.gemv(
                1e-12,
                a,
                y,
                0,
                1,
                1.0,
                y,
                0,
                1,
                scratch = workspace,
            )
        }
    }

    private class DenseLevelThreeFixture {
        val order = 32
        val sides = 8
        val portable = BuiltinEngines.scalar
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 0.25 + (it % 11) * 0.0625 }).also {
            for (i in 0 until order) it.values[i + i * order] = 4.0
        }
        val b = DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { 1.0 })
        val workspace = Scratch()
        init {
            portable.trsm(triangle, b, lower = true, scratch = workspace)
            portable.trmm(triangle, b, lower = true, scratch = workspace)
        }
    }

    private class DenseProductFixture {
        val portable = BuiltinEngines.scalar
        val workspace = Scratch()
        val wide = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48) { 1.0 + (it % 13) * 0.125 })
        val deepLeft = DenseMatrix.wrap(12, 400, DoubleArray(12 * 400) { 1.0 + (it % 7) * 0.25 })
        val deepRight = DenseMatrix.wrap(400, 12, DoubleArray(400 * 12) { 0.5 + (it % 5) * 0.125 })
        val deepTarget = DenseMatrix.wrap(12, 12, DoubleArray(144))
        val small = DenseMatrix.wrap(5, 5, DoubleArray(25) { 1.0 + it })
        val smallTarget = DenseMatrix.wrap(5, 5, DoubleArray(25))
        val packedTarget = DenseMatrix.wrap(48, 48, DoubleArray(48 * 48))
        init {
            portable.gemm(1e-12, wide, false, wide, false, 1.0, packedTarget, scratch = workspace)
            portable.gemm(1e-12, deepLeft, false, deepRight, false, 1.0, deepTarget, scratch = workspace)
            portable.gemm(1e-12, small, false, small, false, 1.0, smallTarget, scratch = workspace)
        }
    }

    private class UnpackedProductFixture {
        val order = 64
        val portable = BuiltinEngines.scalar
        val a = DenseMatrix.wrap(order, 2, DoubleArray(order * 2) { 1.0 + (it % 13) * 0.125 })
        val b = DenseMatrix.wrap(2, 2, DoubleArray(4) { 0.5 })
        val c = DenseMatrix.wrap(order, 2, DoubleArray(order * 2))
    }
}
