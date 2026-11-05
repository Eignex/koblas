package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.testutil.allocation.ARRAY_HEADER_BYTES
import com.eignex.koblas.testutil.allocation.AllocationCase
import com.eignex.koblas.testutil.allocation.FLOOR_BYTES
import com.eignex.koblas.testutil.allocation.parameterRows
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/**
 * Sparse operations allocate nothing once warmed. A fresh sparse result allocates its structure, so those
 * cases bound whatever scratch and copies a call takes beside it.
 */
@RunWith(Parameterized::class)
class SparseOpsAllocationTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        private val engine = BuiltinEngines.simd ?: BuiltinEngines.scalar

        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(
            indexedCases() + levelTwoCases() + levelThreeCases() + resultCases(),
        )

        // These indexed operations have no vector body, so the test task measures them rather than the
        // uninstrumented SIMD allocation check.
        private fun indexedCases(): List<AllocationCase> = listOf(
            with(SparseVectorFixture()) {
                AllocationCase("sparse dot against a dense array allocates nothing", 1_000) {
                    engine.dot(
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
                    engine.gather(gathered.indices, 0, gathered.values, 0, gathered.values.size, dense)
                    gathered
                }
            },
            with(IndexedSliceFixture()) {
                AllocationCase("indexed axpy with independent offsets allocates nothing", 1_000) {
                    engine.axpy(indices, INDEX_OFFSET, values, VALUE_OFFSET, ENTRIES, 1e-12, dense)
                    dense
                }
            },
            with(IndexedSliceFixture()) {
                AllocationCase("indexed scatter with independent offsets allocates nothing", 1_000) {
                    engine.scatter(indices, INDEX_OFFSET, values, VALUE_OFFSET, ENTRIES, dense)
                    dense
                }
            },
        )

        private fun levelTwoCases(): List<AllocationCase> = listOf(
            with(AliasedGemvFixture()) {
                AllocationCase("aliased sparse gemv reuses a workspace", 400) {
                    koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, scratch = workspace)
                    coefficients
                }
            },
            with(SparseLevelTwoFixture()) {
                AllocationCase("sparse gemv allocates nothing", 1_000) {
                    engine.gemv(1e-12, sparse, x, 1.0, y, transpose = true)
                    y
                }
            },
            with(SparseLevelTwoFixture()) {
                AllocationCase("sparse symv allocates nothing", 1_000) {
                    engine.symv(1e-12, sparse, x, 1.0, y, lower = true)
                    y
                }
            },
            with(SparseLevelTwoFixture()) {
                AllocationCase("sparse trsv allocates nothing", 1_000) {
                    engine.trsv(sparse, y, lower = true)
                    y
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

        private fun resultCases(): List<AllocationCase> {
            val skewed = SparseMatrix.ofColumns(
                ROWS,
                COLUMNS,
                List(COLUMNS) { j -> List(32) { i -> (i * 4 + j) to (i + 1.0) } },
            )
            val transposeBytes = resultBytes(ROWS, skewed.nnz)
            return listOf(
                AllocationCase("a sparse transpose allocates only result storage", 100, transposeBytes + FLOOR_BYTES) {
                    skewed.transpose()
                },
            )
        }

        /** A CSC result's bytes: three array headers, [columns] + 1 pointers, and an index and value per entry. */
        private fun resultBytes(columns: Int, entries: Int): Int = 3 * ARRAY_HEADER_BYTES +
            (columns + 1) * Int.SIZE_BYTES + entries * (Int.SIZE_BYTES + Double.SIZE_BYTES)

        private const val ROWS = 128
        private const val COLUMNS = 4
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

    private class AliasedGemvFixture {
        val n = 128
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf((j + 1) % n to 1.0 + j % 5) })
        val coefficients = sparse.values
        val workspace = Scratch()
        init {
            koblas.gemv(1e-12, sparse, coefficients, 1.0, coefficients, scratch = workspace)
        }
    }

    private class SparseLevelTwoFixture {
        val n = 128
        val sparse = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j + 1.0)) })
        val x = DoubleArray(n) { it * 0.01 }
        val y = DoubleArray(n)
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
