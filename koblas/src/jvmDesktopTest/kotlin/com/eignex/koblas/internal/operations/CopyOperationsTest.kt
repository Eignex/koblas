package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.testutil.allocation.*
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import kotlin.test.Test

/** Copies reuse compatible destination storage and snapshot only potentially destructive aliases. */
@RunWith(Parameterized::class)
class CopyOperationsTest internal constructor(private val case: AllocationCase) {
    @Test
    fun `warmed calls stay within their allocation budget`() = case.assertWithinBudget()

    companion object {
        @JvmStatic
        @Parameterized.Parameters(name = "{0}")
        fun cases(): Collection<Array<Any>> = parameterRows(vectorCases() + matrixCases() + constructionCases())

        private fun vectorCases(): List<AllocationCase> = listOf(false, true).map { column ->
            val source = sparseMatrix().columnView(1)
            val destination = if (column) sparseMatrix().columnView(2) else source.copy()
            AllocationCase("sparse vector copy with equal pattern and column destination $column", 100) {
                destination.copyFrom(source)
                destination
            }
        }

        private fun matrixCases(): List<AllocationCase> {
            val sparseSource = sparseMatrix()
            val sparseDestination = sparseMatrix()
            val denseDestination = DenseMatrix.zero(ENTRIES, 3)
            val parent = DenseMatrix.wrap(2 * ENTRIES, 3, DoubleArray(6 * ENTRIES) { it * 0.125 })
            val source = parent.view(0, 0, ENTRIES, 3)
            val disjoint = parent.view(ENTRIES, 0, ENTRIES, 3)
            val identical = source.view(0, 0, ENTRIES, 3)
            return listOf(
                AllocationCase("sparse matrix copy with equal pattern", 100) {
                    sparseDestination.copyFrom(sparseSource)
                    sparseDestination
                },
                AllocationCase("sparse matrix copy into distinct dense storage", 100) {
                    denseDestination.copyFrom(sparseSource)
                    denseDestination
                },
                AllocationCase("disjoint matrix views sharing a buffer", 100) {
                    disjoint.copyFrom(source)
                    parent
                },
                AllocationCase("identical matrix views sharing a buffer", 100) {
                    identical.copyFrom(source)
                    parent
                },
            )
        }

        private fun constructionCases(): List<AllocationCase> = listOf(
            "vector copy",
            "matrix copy",
            "dense axpy",
            "sparse axpy",
        ).map { operation ->
            val source = DenseVector.wrap(DoubleArray(ENTRIES) { it + 1.0 })
            val destination = SparseVector.wrap(ENTRIES, IntArray(0), DoubleArray(0))
            val sparseSource = SparseVector.wrap(ENTRIES, IntArray(ENTRIES) { it }, source.values)
            val matrixSource = DenseMatrix.wrap(16, ENTRIES / 16, source.values)
            val matrixDestination = SparseMatrix.ofColumns(16, ENTRIES / 16, List(ENTRIES / 16) { emptyList() })
            // Doubling primitive buffers through this power of two costs less than two final array pairs.
            val budget = 2.0 * ENTRIES * (Int.SIZE_BYTES + Double.SIZE_BYTES) + 512
            AllocationCase("$operation stores primitive entries", 100, budget) {
                when (operation) {
                    "vector copy" -> destination.copyFrom(source)
                    "matrix copy" -> matrixDestination.copyFrom(matrixSource)
                    "dense axpy" -> KoblasEngine.scalar.axpy(1e-12, source, destination)
                    "sparse axpy" -> KoblasEngine.scalar.axpy(1e-12, sparseSource, destination)
                }
                if (operation == "matrix copy") matrixDestination else destination
            }
        }

        private fun sparseMatrix(): SparseMatrix = SparseMatrix.wrap(
            ENTRIES,
            3,
            intArrayOf(0, ENTRIES, 2 * ENTRIES, 3 * ENTRIES),
            IntArray(3 * ENTRIES) { it % ENTRIES },
            DoubleArray(3 * ENTRIES) { it * 0.125 },
        )

        private const val ENTRIES = 512
    }
}
