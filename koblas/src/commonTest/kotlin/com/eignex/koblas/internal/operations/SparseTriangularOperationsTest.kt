package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.dense.withEngines
import kotlin.test.Test

class SparseTriangularOperationsTest {
    @Test
    fun `every triangular alias variant agrees with the reference`() = withEngines { engine ->
        for (right in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unit in booleanArrayOf(false, true)) {
                        for (solve in booleanArrayOf(false, true)) {
                            assertTriangularAgreesWithReference(engine, right, lower, transpose, unit, solve)
                        }
                    }
                }
            }
        }
    }

    private fun assertTriangularAgreesWithReference(
        engine: KoblasEngine,
        right: Boolean,
        lower: Boolean,
        transpose: Boolean,
        unit: Boolean,
        solve: Boolean,
    ) {
        val order = 4
        val values = DoubleArray(order * order) { i -> if (i % (order + 1) == 0) 4.0 else i * 0.02 }
        val a = SparseMatrix.wrap(
            order,
            order,
            IntArray(order + 1) { it * order },
            IntArray(values.size) { it % order },
            values,
        )
        val destination = DenseMatrix.wrap(
            if (right) 3 else order,
            if (right) order else 3,
            values,
            1,
            if (right) order else order + 1,
        )
        val backing = values.copyOf()
        val expected = DenseMatrix.ofRows(destination.toArray())
        val reference = DenseMatrix.ofRows(a.toArray())
        if (solve) {
            KoblasEngine.scalar.trsm(reference, expected, lower, transpose, unit, right, 0.5)
            engine.trsm(a, destination, lower, transpose, unit, right, 0.5, Scratch())
        } else {
            KoblasEngine.scalar.trmm(reference, expected, lower, transpose, unit, right, 0.5)
            engine.trmm(a, destination, lower, transpose, unit, right, 0.5, Scratch())
        }
        for (j in 0 until destination.cols) {
            for (i in 0 until destination.rows) {
                backing[destination.offset + i + j * destination.leadingDimension] = expected[i, j]
            }
        }
        assertClose(backing, values, "${engine.name} triangular aliases")
    }
}
