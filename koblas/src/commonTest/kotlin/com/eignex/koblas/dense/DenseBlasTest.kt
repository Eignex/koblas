package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DenseBlasTest {
    private fun engines() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd, koblas).distinct()

    @Test
    fun `windowed GER agrees with the reference with independent offsets and parent strides`() {
        for (engine in engines()) {
            for ((rows, columns) in listOf(1 to 1, 3 to 2, 5 to 4)) {
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                val x = DoubleArray(10) { (it - 4) / 16.0 }
                val y = DoubleArray(11) { (5 - it) / 8.0 }
                val originalX = x.copyOf()
                val originalY = y.copyOf()
                val expected = referenceWindow(a, x, y, rows, columns)

                engine.ger(-0.875, x, y, a, 2, 3, rows, columns, 1, 4)

                assertWindowAgreesWithReference(expected, a, rows, columns)
                assertContentEquals(originalX, x)
                assertContentEquals(originalY, y)
            }
        }
    }

    @Test
    fun `windowed GER snapshots either or both aliased vector ranges`() {
        for (engine in engines()) {
            for (alias in 1..3) {
                val workspace = Workspace()
                repeat(2) {
                    val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { (it % 13 - 6) / 8.0 })
                    val x = if (alias != 2) a.values else DoubleArray(10) { (it - 4) / 16.0 }
                    val y = if (alias != 1) a.values else DoubleArray(11) { (5 - it) / 8.0 }
                    val xOffset = if (alias != 2) 29 else 1
                    val yOffset = if (alias != 1) 30 else 4
                    val expected = referenceWindow(a, x, y, 3, 2, xOffset, yOffset)

                    engine.ger(-0.875, x, y, a, 2, 3, 3, 2, xOffset, yOffset, workspace)

                    assertWindowAgreesWithReference(expected, a, 3, 2)
                    assertEquals(if (alias != 2) 1 else 0, workspace.available(3))
                    assertEquals(if (alias != 1) 1 else 0, workspace.available(2))
                    assertEquals(0, workspace.available(a.values.size))
                }
            }
        }
    }

    @Test
    fun `windowed GER validates every range before writing even with zero alpha`() {
        val valid = intArrayOf(2, 3, 3, 2, 1, 4)
        for (engine in engines()) {
            for (alpha in doubleArrayOf(0.0, -1.0)) {
                for (index in valid.indices) {
                    for (invalid in intArrayOf(-1, Int.MAX_VALUE)) {
                        val args = valid.copyOf().also { it[index] = invalid }
                        val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { it.toDouble() })
                        val before = a.values.copyOf()

                        assertFailsWith<DimensionMismatch> {
                            engine.ger(
                                alpha, DoubleArray(10), DoubleArray(11), a,
                                args[0], args[1], args[2], args[3], args[4], args[5],
                            )
                        }

                        assertContentEquals(before, a.values)
                    }
                }
            }
        }
    }

    @Test
    fun `empty windows and zero alpha do not read entries or borrow scratch`() {
        for (engine in engines()) {
            for (shape in listOf(0 to 0, 0 to 2, 3 to 0, 3 to 2)) {
                val a = DenseMatrix.wrap(9, 8, DoubleArray(72) { Double.NaN })
                val workspace = Workspace()
                val alpha = if (shape.first > 0 && shape.second > 0) 0.0 else Double.NaN

                engine.ger(
                    alpha, a.values, a.values, a,
                    9 - shape.first, 8 - shape.second, shape.first, shape.second,
                    72 - shape.first, 72 - shape.second, workspace,
                )

                assertContentEquals(DoubleArray(72) { Double.NaN }, a.values)
                assertEquals(0, workspace.idleLengths())
            }
        }
    }

    @Test
    fun `windowed GER agrees with the reference on shrinking supports and extreme finite inputs`() {
        for (engine in engines()) {
            assertShrinkingGerAgreesWithReference { x, y, a ->
                val start = x.indexOfFirst { it != 0.0 }
                val extent = a.rows - start
                engine.ger(-1.0, x, y, a, start, start, extent, extent, start, start)
            }
        }
    }

    @Suppress("LongParameterList") // the rectangle and independently selected input ranges
    private fun referenceWindow(
        a: DenseMatrix,
        x: DoubleArray,
        y: DoubleArray,
        rows: Int,
        columns: Int,
        xOffset: Int = 1,
        yOffset: Int = 4,
    ): DenseMatrix {
        val expected = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())
        val rectangle = DenseMatrix.wrap(
            rows,
            columns,
            DoubleArray(rows * columns) { index ->
                a[2 + index % rows, 3 + index / rows]
            },
        )
        ReferenceBlas.ger(
            -0.875,
            x.copyOfRange(xOffset, xOffset + rows),
            y.copyOfRange(yOffset, yOffset + columns),
            rectangle,
        )
        for (j in 0 until columns) for (i in 0 until rows) expected[2 + i, 3 + j] = rectangle[i, j]
        return expected
    }

    private fun assertWindowAgreesWithReference(expected: DenseMatrix, actual: DenseMatrix, rows: Int, columns: Int) {
        assertClose(expected, actual, "windowed GER")
        for (j in 0 until actual.cols) {
            for (i in 0 until actual.rows) {
                if (i !in 2 until 2 + rows || j !in 3 until 3 + columns) {
                    assertEquals(expected[i, j], actual[i, j], "outside rectangle at $i $j")
                }
            }
        }
    }
}
