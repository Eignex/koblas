package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.engine.BuiltinEngines
import com.eignex.koblas.internal.operations.sparse.withSymmetricRankScratch
import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@OptIn(UnsafeKoblasApi::class)
class SparseRankUpdateTest {

    @Test
    fun `sparse syrk forms products of stored entries only`() {
        val exceptional = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY, 1 to 0.0)))
        val implicit = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val implicitDense = DenseMatrix.zero(2)

        val stored = DenseMatrix.zero(2)
        koblas.syrk(1.0, exceptional, false, 0.0, stored)
        koblas.syrk(1.0, implicit, false, 0.0, implicitDense)

        assertTrue(stored[1, 0].isNaN(), "a stored zero must form zero times infinity")
        assertEquals(0.0, implicitDense[1, 0], "implicit sparse zero must not form zero times infinity")
    }

    @Test
    fun `sparse syrk agrees with the independent sparse oracle`() {
        val a = SparseMatrix.ofColumns(
            4,
            4,
            listOf(
                listOf(0 to 2.0, 2 to -1.5),
                listOf(1 to 3.25, 3 to 0.5),
                listOf(2 to 4.0),
                listOf(3 to -2.25),
            ),
        )
        for (lower in booleanArrayOf(false, true)) {
            val expected = DenseMatrix.wrap(4, 4, DoubleArray(16) { 0.25 * it })
            val actual = expected.copyOf()
            ReferenceSparseBlas.syrk(1.25, a, false, -0.5, expected, lower)

            koblas.syrk(1.25, a, false, -0.5, actual, lower)

            assertContentEquals(expected.values, actual.values, "lower=$lower")
        }
    }

    @Test
    fun `dense syrk reuses dirty scratch across different patterns and triangle flags`() {
        val patterns = listOf(
            SparseMatrix.ofColumns(
                5,
                4,
                listOf(
                    listOf(0 to 2.0, 2 to -1.0),
                    listOf(1 to 3.0, 4 to 0.0),
                    emptyList(),
                    listOf(0 to -2.0, 3 to 4.0),
                ),
            ),
            SparseMatrix.ofColumns(
                5,
                4,
                listOf(
                    listOf(1 to -1.0, 3 to 2.0),
                    emptyList(),
                    listOf(0 to 0.0, 2 to 3.0, 4 to -2.0),
                    listOf(2 to 1.0),
                ),
            ),
        )
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
            val workspace = Scratch()
            // Epoch one collides with the first column unless marks are cleared; nonzero row counts
            // expose an adjacency builder that assumes its borrowed pointers are fresh.
            for (transpose in booleanArrayOf(false, true)) {
                for (a in patterns) {
                    val order = if (transpose) a.cols else a.rows
                    withSymmetricRankScratch(workspace, order, a.rows, a.nnz) {
                            sums,
                            marks,
                            touched,
                            pointers,
                            columns,
                            positions,
                            cursor,
                        ->
                        sums.fill(Double.NaN)
                        for (buffer in listOf(marks, touched, pointers, columns, positions, cursor)) buffer.fill(1)
                    }
                }
            }
            repeat(2) { pass ->
                for (transpose in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(true, false)) {
                        for ((pattern, a) in patterns.withIndex()) {
                            val order = if (transpose) a.cols else a.rows
                            val initial = DenseMatrix.wrap(order, order, DoubleArray(order * order) { -1000.0 - it })
                            for (j in 0 until order) {
                                for (i in 0 until order) {
                                    if (if (lower) i >= j else i <= j) initial[i, j] = (i - j) * 0.25
                                }
                            }
                            val expected = initial.copyOf()
                            val actual = initial.copyOf()
                            ReferenceSparseBlas.syrk(0.875, a, transpose, -0.25, expected, lower)

                            engine.syrk(0.875, a, transpose, -0.25, actual, lower, scratch = workspace)

                            val context = "${engine.name} pass=$pass pattern=$pattern transpose=$transpose lower=$lower"
                            assertClose(expected, actual, context)
                            for (j in 0 until order) {
                                for (i in 0 until order) {
                                    if (if (lower) i < j else i > j) assertEquals(initial[i, j], actual[i, j], context)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
