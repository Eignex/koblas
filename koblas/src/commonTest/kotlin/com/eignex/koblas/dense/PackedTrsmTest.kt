package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PackedTrsmTest {
    @Test
    fun `the portable packed solve agrees with a written out right product`() {
        val tileRows = 5
        val tileColumns = 4
        val rng = Random(20260908)
        for (validRows in 0..tileRows) {
            for (order in 0..tileColumns) {
                for (lower in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val triangle = packedTriangle(tileColumns, order, lower, unitDiag, rng)
                        val expected = DoubleArray(tileRows * tileColumns) { -91.0 }
                        for (column in 0 until order) {
                            for (row in 0 until validRows) expected[row + column * tileRows] = rng.nextDouble(-2.0, 2.0)
                        }
                        val actual = rightProduct(
                            tileRows, tileColumns, validRows, order, triangle, lower, unitDiag, expected,
                        )

                        portableTrsmTile(
                            tileRows, tileColumns, validRows, order, triangle, 0, lower, unitDiag, actual, 0,
                        )

                        for (column in 0 until order) {
                            assertClose(
                                expected.copyOfRange(column * tileRows, column * tileRows + validRows),
                                actual.copyOfRange(column * tileRows, column * tileRows + validRows),
                                "rows=$validRows order=$order lower=$lower unit=$unitDiag column=$column",
                            )
                            for (row in validRows until tileRows) {
                                assertEquals(-91.0, actual[row + column * tileRows], "row padding changed")
                            }
                        }
                        for (column in order until tileColumns) {
                            for (row in 0 until tileRows) {
                                assertEquals(-91.0, actual[row + column * tileRows], "column padding changed")
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `the packed solve skips zero triangle coefficients`() {
        val triangle = doubleArrayOf(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        val x = doubleArrayOf(Double.POSITIVE_INFINITY, 1.0)

        portableTrsmTile(1, 4, 1, 2, triangle, 0, lower = true, unitDiag = false, x, 0)

        assertEquals(Double.POSITIVE_INFINITY, x[0])
        assertEquals(1.0, x[1])
    }

    @Test
    fun `the packed solve divides by singular and subnormal diagonals`() {
        val singular = doubleArrayOf(0.0, 0.0, 0.0, 0.0)
        val singularResult = doubleArrayOf(1.0)
        portableTrsmTile(1, 4, 1, 1, singular, 0, lower = true, unitDiag = false, singularResult, 0)
        assertTrue(singularResult[0].isInfinite())

        val subnormal = doubleArrayOf(Double.MIN_VALUE, 0.0, 0.0, 0.0)
        val finiteResult = doubleArrayOf(Double.MIN_VALUE)
        portableTrsmTile(1, 4, 1, 1, subnormal, 0, lower = true, unitDiag = false, finiteResult, 0)
        assertEquals(1.0, finiteResult[0], "a reciprocal would overflow before forming this finite quotient")
    }

    @Test
    fun `a unit diagonal is not read`() {
        val triangle = doubleArrayOf(Double.NaN, 0.0, 0.0, 0.0)
        val x = doubleArrayOf(3.0)

        portableTrsmTile(1, 4, 1, 1, triangle, 0, lower = true, unitDiag = true, x, 0)

        assertEquals(3.0, x[0])
    }

    private fun packedTriangle(
        tileColumns: Int,
        order: Int,
        lower: Boolean,
        unitDiag: Boolean,
        rng: Random,
    ): DoubleArray = DoubleArray(tileColumns * tileColumns) { index ->
        val row = index / tileColumns
        val column = index % tileColumns
        when {
            row >= order || column >= order -> 0.0
            row == column && unitDiag -> Double.NaN
            row == column -> rng.nextDouble(0.5, 2.0)
            lower && row > column -> rng.nextDouble(-1.0, 1.0)
            !lower && row < column -> rng.nextDouble(-1.0, 1.0)
            else -> Double.NaN
        }
    }

    @Suppress("LongParameterList")
    private fun rightProduct(
        tileRows: Int,
        tileColumns: Int,
        validRows: Int,
        order: Int,
        triangle: DoubleArray,
        lower: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
    ): DoubleArray = DoubleArray(tileRows * tileColumns) { index ->
        val row = index % tileRows
        val column = index / tileRows
        if (row >= validRows || column >= order) {
            -91.0
        } else {
            var sum = 0.0
            for (inner in 0 until order) {
                val stored = if (lower) inner >= column else inner <= column
                if (stored) {
                    val coefficient = if (unitDiag && inner == column) {
                        1.0
                    } else {
                        triangle[inner * tileColumns + column]
                    }
                    sum += x[row + inner * tileRows] * coefficient
                }
            }
            sum
        }
    }
}
