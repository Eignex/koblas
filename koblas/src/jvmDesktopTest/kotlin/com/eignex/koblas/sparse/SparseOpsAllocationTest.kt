package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.testutil.allocation.*
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/**
 * Selected indexed kernels and the shared Level 3 schedules retain their allocation budgets. Portable
 * Level 2 traversal and sparse construction are checked for correctness and scratch reuse instead.
 */
@RunWith(Parameterized::class)
class SparseOpsAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        private val engine = KoblasEngine.simd ?: KoblasEngine.scalar

        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(
            indexedCases() + reductionCases() + levelThreeCases(),
        )

        // These indexed operations have no vector body, so the test task measures them rather than the
        // uninstrumented SIMD allocation check.
        private fun indexedCases(): List<AllocationCase> = listOf(
            with(SparseVectorFixture()) {
                AllocationCase("sparse dot against a dense array allocates nothing", 1_000) {
                    engine.kernels.dot(
                        vector.indices,
                        0,
                        vector.values,
                        0,
                        vector.values.size,
                        x,
                    )
                }
            },
            with(IndexedSliceFixture()) {
                AllocationCase("indexed gather allocates nothing", 1_000) {
                    engine.kernels.gather(gathered.indices, 0, gathered.values, 0, gathered.values.size, dense)
                    gathered
                }
            },
            with(IndexedSliceFixture()) {
                AllocationCase("indexed axpy with independent offsets allocates nothing", 1_000) {
                    engine.kernels.axpy(indices, INDEX_OFFSET, values, VALUE_OFFSET, ENTRIES, 1e-12, dense)
                    dense
                }
            },
            with(IndexedSliceFixture()) {
                AllocationCase("indexed scatter with independent offsets allocates nothing", 1_000) {
                    engine.kernels.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, ENTRIES, dense)
                    dense
                }
            },
        )

        private fun levelThreeCases(): List<AllocationCase> = listOf(
            with(SparseLevelThreeFixture()) {
                AllocationCase("left sparse gemm reuses a workspace", 300) {
                    engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, scratch = workspace)
                    panel
                }
            },
            with(SparseLevelThreeFixture()) {
                AllocationCase("right sparse gemm reuses a workspace", 300) {
                    engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, scratch = workspace)
                    transposed
                }
            },
            with(SparseLevelThreeFixture()) {
                AllocationCase("sparse syrk reuses a workspace", 300) {
                    engine.syrk(1e-12, sparse, false, 1.0, square, scratch = workspace)
                    square
                }
            },
            with(SparseLevelThreeFixture()) {
                AllocationCase("sparse symm reuses a workspace", 300) {
                    engine.symm(1e-12, sparse, b, 1.0, panel, scratch = workspace)
                    panel
                }
            },
            with(SparseLevelThreeFixture()) {
                AllocationCase("sparse trsm reuses a workspace", 300) {
                    engine.trsm(sparse, solveBlock, lower = true, scratch = workspace)
                    solveBlock
                }
            },
            with(SparseLevelThreeFixture()) {
                AllocationCase("sparse trmm reuses a workspace", 300) {
                    engine.trmm(sparse, multiplyBlock, lower = true, scratch = workspace)
                    multiplyBlock
                }
            },
        )

        private fun reductionCases(): List<AllocationCase> = listOf(8, 256).map { columns ->
            val matrix = SparseMatrix.ofColumns(
                Int.MAX_VALUE,
                columns,
                List(columns) { listOf(257 to 1.0, Int.MAX_VALUE - 1 to -0.5) },
            )
            AllocationCase("sparse infinity norm retains two row sums across $columns columns", 100, budget = 1024.0) {
                matrix.norm(Norm.Infinity)
            }
        }

        private const val ENTRIES = 512
        private const val INDEX_OFFSET = 3
        private const val VALUE_OFFSET = 5
    }

    private class SparseVectorFixture {
        val n = 128
        val vector = SparseVector.of(n, IntArray(n / 2) { it * 2 }, DoubleArray(n / 2) { it + 1.0 })
        val x = DoubleArray(n) { it * 0.01 }
    }

    /** Stored runs at independent offsets into padded arrays, over a dense operand four times as long. */
    private class IndexedSliceFixture {
        val dimension = 4 * ENTRIES
        val indices = IntArray(ENTRIES + 7) {
            if (it in INDEX_OFFSET until INDEX_OFFSET + ENTRIES) 1 + (it - INDEX_OFFSET) * 4 else -1
        }
        val values = DoubleArray(ENTRIES + 9) {
            if (it in VALUE_OFFSET until VALUE_OFFSET + ENTRIES) 0.5 + (it - VALUE_OFFSET) * 0.125 else Double.NaN
        }
        val dense = DoubleArray(dimension) { 1.0 + (it % 17) * 0.03125 }
        val gathered = SparseVector.wrap(dimension, IntArray(ENTRIES) { 1 + it * 4 }, DoubleArray(ENTRIES))
    }

    private class SparseLevelThreeFixture {
        val n = 64
        val rightHandSides = 8
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val b = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val solveBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val multiplyBlock = DenseMatrix.zero(n, rightHandSides).also { it.values.fill(1.0) }
        val transposed = DenseMatrix.zero(rightHandSides, n)
        val square = DenseMatrix.zero(n, n)
        val panel = DenseMatrix.zero(n, rightHandSides)
        val workspace = Scratch()
        init {
            // One warm call per shape, so the loans exist before the measured loop asks for them again.
            engine.gemm(1e-12, sparse, false, b, false, 1.0, panel, scratch = workspace)
            engine.gemm(1e-12, sparse, false, b, true, 1.0, transposed, right = true, scratch = workspace)
            engine.syrk(1e-12, sparse, false, 1.0, square, scratch = workspace)
            engine.symm(1e-12, sparse, b, 1.0, panel, scratch = workspace)
            engine.trsm(sparse, solveBlock, lower = true, scratch = workspace)
            engine.trmm(sparse, multiplyBlock, lower = true, scratch = workspace)
        }
    }
}
