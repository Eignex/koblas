package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class PackedTrsmSimdTest {
    @Test
    fun `the SIMD packed solve agrees with the portable tile`() {
        if (!simdAvailable) return
        val tileRows = Simd.tileRows
        val tileColumns = Simd.TILE_COLS
        val rng = Random(20260909)
        for (validRows in 0..tileRows) {
            for (order in 0..tileColumns) {
                for (lower in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val triangle = DoubleArray(tileColumns * tileColumns) { index ->
                            val row = index / tileColumns
                            val column = index % tileColumns
                            when {
                                row == column && unitDiag -> Double.NaN
                                row == column -> rng.nextDouble(0.5, 2.0)
                                lower && row > column -> rng.nextDouble(-1.0, 1.0)
                                !lower && row < column -> rng.nextDouble(-1.0, 1.0)
                                else -> 0.0
                            }
                        }
                        val expected = DoubleArray(tileRows * tileColumns) { rng.nextDouble(-2.0, 2.0) }
                        val actual = expected.copyOf()
                        portableTrsmTile(
                            tileRows, tileColumns, validRows, order, triangle, 0,
                            lower, unitDiag, expected, 0,
                        )

                        simdTrsmTile(validRows, order, triangle, 0, lower, unitDiag, actual, 0)

                        assertClose(
                            expected,
                            actual,
                            "rows=$validRows order=$order lower=$lower unit=$unitDiag",
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `the SIMD packed solve skips zero coefficients with nonfinite values`() {
        if (!simdAvailable) return
        val rows = Simd.tileRows
        val triangle = DoubleArray(Simd.TILE_COLS * Simd.TILE_COLS).also {
            it[0] = 1.0
            it[Simd.TILE_COLS + 1] = 1.0
        }
        val x = DoubleArray(rows * Simd.TILE_COLS)
        x[0] = Double.POSITIVE_INFINITY
        x[rows] = 1.0

        simdTrsmTile(1, 2, triangle, 0, lower = true, unitDiag = false, x, 0)

        assertEquals(Double.POSITIVE_INFINITY, x[0])
        assertEquals(1.0, x[rows])
    }

    @Test
    fun `the fused SIMD tile agrees with update then solve`() {
        if (!simdAvailable) return
        val rows = Simd.tileRows
        val columns = Simd.TILE_COLS
        val rng = Random(20260910)
        for (depth in intArrayOf(0, 1, 9)) {
            for (validRows in intArrayOf(1, rows - 1, rows)) {
                for (order in 1..columns) {
                    for (lower in booleanArrayOf(false, true)) {
                        val packedA = DoubleArray(depth * rows) { rng.nextDouble(-1.0, 1.0) }
                        val packedB = DoubleArray(depth * columns) { rng.nextDouble(-1.0, 1.0) }
                        val triangle = DoubleArray(columns * columns) { index ->
                            val row = index / columns
                            val column = index % columns
                            when {
                                row == column -> rng.nextDouble(0.5, 2.0)
                                lower && row > column -> rng.nextDouble(-1.0, 1.0)
                                !lower && row < column -> rng.nextDouble(-1.0, 1.0)
                                else -> 0.0
                            }
                        }
                        val expected = DoubleArray(rows * columns) { rng.nextDouble(-1.0, 1.0) }
                        val actual = expected.copyOf()
                        Simd.gemmTile(depth, packedA, 0, packedB, 0, expected, 0, rows)
                        simdTrsmTile(validRows, order, triangle, 0, lower, false, expected, 0)

                        simdGemmTrsmTile(
                            depth, validRows, order, packedA, 0, packedB, 0,
                            triangle, 0, lower, false, actual, 0,
                        )

                        assertClose(
                            expected,
                            actual,
                            "depth=$depth rows=$validRows order=$order lower=$lower",
                        )
                    }
                }
            }
        }
    }
}
