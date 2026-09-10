package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertTrue

class PackedWorkTest {
    @Test
    fun `packed solve fixtures agree with the scalar equation`() {
        val engines = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.c, BuiltinEngines.simd).distinct()
        for (engine in engines) {
            val kernels = engine.packedKernels
            for (rows in listOf(kernels.gemmTileRows, max(1, kernels.gemmTileRows - 1))) {
                for (order in listOf(kernels.gemmTileCols, max(1, kernels.gemmTileCols - 2))) {
                    for (lower in listOf(false, true)) {
                        for (unit in listOf(false, true)) {
                            assertPackedCase(kernels, rows, order, lower, unit, depth = 0)
                            assertPackedCase(kernels, rows, order, lower, unit, depth = 3)
                        }
                    }
                }
            }
        }
    }

    @Suppress("LongParameterList")
    private fun assertPackedCase(
        kernels: com.eignex.koblas.dense.PackedKernels,
        rows: Int,
        order: Int,
        lower: Boolean,
        unit: Boolean,
        depth: Int,
    ) {
        val tileRows = kernels.gemmTileRows
        val tileColumns = kernels.gemmTileCols
        val triangle = packedTriangleFixture(order, tileColumns, lower)
        val left = Fixtures.vector(tileRows * depth, 1)
        val right = Fixtures.vector(tileColumns * depth, 2)
        val actual = Fixtures.vector(tileRows * tileColumns, 4)
        val expected = actual.copyOf()
        if (depth > 0) {
            for (column in 0 until order) for (row in 0 until rows) {
                var product = 0.0
                for (inner in 0 until depth) {
                    product += left[row + inner * tileRows] * right[column + inner * tileColumns]
                }
                expected[row + column * tileRows] -= product
            }
        }
        solveRight(expected, tileRows, rows, order, triangle, tileColumns, lower, unit)

        if (depth == 0) {
            kernels.trsmTile(rows, order, triangle, 0, lower, unit, actual, 0)
        } else {
            kernels.gemmTrsmTile(depth, rows, order, left, 0, right, 0, triangle, 0, lower, unit, actual, 0)
        }

        for (index in expected.indices) {
            val tolerance = 2e-12 * (1.0 + abs(expected[index]))
            assertTrue(abs(expected[index] - actual[index]) <= tolerance, "index=$index rows=$rows order=$order lower=$lower unit=$unit depth=$depth")
        }
    }

    @Suppress("LongParameterList")
    private fun solveRight(
        values: DoubleArray,
        tileRows: Int,
        rows: Int,
        order: Int,
        triangle: DoubleArray,
        tileColumns: Int,
        lower: Boolean,
        unit: Boolean,
    ) {
        val columns = if (lower) (order - 1 downTo 0) else (0 until order)
        for (column in columns) for (row in 0 until rows) {
            var value = values[row + column * tileRows]
            val solved = if (lower) (column + 1 until order) else (0 until column)
            for (inner in solved) value -= values[row + inner * tileRows] * triangle[inner * tileColumns + column]
            values[row + column * tileRows] = if (unit) value else value / triangle[column * tileColumns + column]
        }
    }
}
