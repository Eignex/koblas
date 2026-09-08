package com.eignex.koblas.dense

import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.min
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class F64PackedPanelsTest {
    @Test
    fun `left panels round trip across a partial edge`() {
        val tile = F64PackedPanels.tileRows
        val rows = tile + 1
        val depth = 3
        val source = matrix(rows + 2, depth + 2)
        val packed = DoubleArray(F64PackedPanels.leftSize(rows, depth) + 4) { Double.NaN }
        val restored = F64DenseMatrix.zero(rows + 2, depth + 2)

        F64PackedPanels.packLeft(
            source,
            packed,
            rows,
            depth,
            sourceRow = 1,
            sourceColumn = 1,
            alpha = 2.0,
            destinationOffset = 2,
        )
        F64PackedPanels.writeLeft(
            packed,
            restored,
            rows,
            depth,
            sourceOffset = 2,
            destinationRow = 1,
            destinationColumn = 1,
        )

        for (j in 0 until depth) {
            for (i in 0 until rows) assertEquals(2.0 * source[i + 1, j + 1], restored[i + 1, j + 1])
        }
        val edgePanel = 2 + tile * depth
        for (step in 0 until depth) {
            for (lane in 1 until tile) assertPositiveZero(packed[edgePanel + step * tile + lane])
        }
        assertTrue(packed[0].isNaN())
        assertTrue(packed[packed.lastIndex].isNaN())
    }

    @Test
    fun `right panels transpose and round trip`() {
        val depth = 3
        val columns = F64PackedPanels.tileColumns + 1
        val source = matrix(columns + 2, depth + 2)
        val packed = DoubleArray(F64PackedPanels.rightSize(depth, columns))
        val restored = F64DenseMatrix.zero(columns + 2, depth + 2)

        F64PackedPanels.packRight(
            source,
            packed,
            depth,
            columns,
            sourceRow = 1,
            sourceColumn = 1,
            transpose = true,
        )
        F64PackedPanels.writeRight(
            packed,
            restored,
            depth,
            columns,
            destinationRow = 1,
            destinationColumn = 1,
            transpose = true,
        )

        for (j in 0 until columns) {
            for (i in 0 until depth) assertEquals(source[j + 1, i + 1], restored[j + 1, i + 1])
        }
    }

    @Test
    fun `symmetric packing never reads the poisoned triangle`() {
        val order = min(F64PackedPanels.tileRows, F64PackedPanels.tileColumns) + 1
        val source = F64DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) source[i, j] = if (i >= j) 100.0 * j + i else Double.NaN
        }
        val left = DoubleArray(F64PackedPanels.leftSize(order, order))
        val right = DoubleArray(F64PackedPanels.rightSize(order, order))
        val leftRestored = F64DenseMatrix.zero(order)
        val rightRestored = F64DenseMatrix.zero(order)

        F64PackedPanels.packSymmetricLeft(source, left, order, order, lower = true)
        F64PackedPanels.packSymmetricRight(source, right, order, order, lower = true)
        F64PackedPanels.writeLeft(left, leftRestored, order, order)
        F64PackedPanels.writeRight(right, rightRestored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) {
                val expected = if (i >= j) source[i, j] else source[j, i]
                assertEquals(expected, leftRestored[i, j])
                assertEquals(expected, rightRestored[i, j])
            }
        }
    }

    @Test
    fun `triangular unit diagonal is materialized without reading`() {
        val order = min(F64PackedPanels.tileColumns, 5)
        val source = F64DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) {
                source[i, j] = when {
                    i > j -> 10.0 * j + i
                    else -> Double.NaN
                }
            }
        }
        val packed = DoubleArray(F64PackedPanels.rightSize(order, order))
        val restored = F64DenseMatrix.zero(order)

        F64PackedPanels.packTriangularRight(
            source,
            packed,
            order,
            order,
            lower = true,
            transpose = true,
            unitDiagonal = true,
        )
        F64PackedPanels.writeRight(packed, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) {
                val expected = when {
                    i == j -> 1.0
                    i < j -> source[j, i]
                    else -> 0.0
                }
                assertEquals(expected, restored[i, j], "entry ($i, $j)")
            }
        }
    }

    @Test
    fun `left alpha preserves exceptional products and zero padding`() {
        val rows = F64PackedPanels.tileRows - 1
        val source = F64DenseMatrix.zero(rows, 1)
        val packed = DoubleArray(F64PackedPanels.leftSize(rows, 1))

        F64PackedPanels.packLeft(source, packed, rows, 1, alpha = Double.POSITIVE_INFINITY)

        for (lane in 0 until rows) assertTrue(packed[lane].isNaN())
        for (lane in rows until F64PackedPanels.tileRows) assertPositiveZero(packed[lane])
    }

    @Test
    fun `triangular structural zero survives infinite alpha`() {
        val order = min(F64PackedPanels.tileRows, 3)
        val source = F64DenseMatrix.zero(order)
        val packed = DoubleArray(F64PackedPanels.leftSize(order, order))
        val restored = F64DenseMatrix.zero(order)

        F64PackedPanels.packTriangularLeft(
            source,
            packed,
            order,
            order,
            lower = true,
            alpha = Double.POSITIVE_INFINITY,
        )
        F64PackedPanels.writeLeft(packed, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) {
                if (i < j) assertPositiveZero(restored[i, j]) else assertTrue(restored[i, j].isNaN())
            }
        }
    }

    @Test
    fun `same backing packing uses workspace staging`() {
        val order = F64PackedPanels.tileRows
        val source = matrix(order, order)
        val expected = source.toArray()
        val workspace = Workspace().also { it.reserve(source.data.size, 1) }

        F64PackedPanels.packLeft(
            source,
            source.data,
            order,
            order,
            transpose = true,
            workspace = workspace,
        )
        val restored = F64DenseMatrix.zero(order)
        F64PackedPanels.writeLeft(source.data, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) assertEquals(expected[j][i], restored[i, j])
        }
    }

    @Test
    fun `same backing writeback uses workspace staging`() {
        val order = F64PackedPanels.tileColumns
        val source = matrix(order, order)
        val packed = DoubleArray(F64PackedPanels.rightSize(order, order))
        F64PackedPanels.packRight(source, packed, order, order)
        val destination = F64DenseMatrix.wrap(order, order, packed.copyOf())
        val workspace = Workspace().also { it.reserve(packed.size, 1) }

        F64PackedPanels.writeRight(
            destination.data,
            destination,
            order,
            order,
            workspace = workspace,
        )

        assertEquals(source, destination)
    }

    @Test
    fun `padding restoration leaves valid entries unchanged`() {
        val rows = F64PackedPanels.tileRows + 1
        val depth = 2
        val panel = DoubleArray(F64PackedPanels.leftSize(rows, depth)) { it + 1.0 }
        val before = panel.copyOf()

        F64PackedPanels.clearLeftPadding(panel, rows, depth)

        val edgePanel = F64PackedPanels.tileRows * depth
        for (step in 0 until depth) {
            assertEquals(
                before[edgePanel + step * F64PackedPanels.tileRows],
                panel[edgePanel + step * F64PackedPanels.tileRows],
            )
            for (lane in 1 until F64PackedPanels.tileRows) {
                assertPositiveZero(panel[edgePanel + step * F64PackedPanels.tileRows + lane])
            }
        }
    }

    @Test
    fun `packed panels feed the platform tile`() {
        val rows = F64PackedPanels.tileRows - 1
        val columns = F64PackedPanels.tileColumns - 1
        val depth = 3
        val a = matrix(rows, depth)
        val b = matrix(depth, columns)
        val packedA = DoubleArray(F64PackedPanels.leftSize(rows, depth))
        val packedB = DoubleArray(F64PackedPanels.rightSize(depth, columns))
        val c = DoubleArray(F64PackedPanels.tileRows * F64PackedPanels.tileColumns)

        F64PackedPanels.packLeft(a, packedA, rows, depth)
        F64PackedPanels.packRight(b, packedB, depth, columns)
        F64PlatformKernels.gemmTile(
            depth,
            packedA,
            0,
            packedB,
            0,
            c,
            0,
            F64PackedPanels.tileRows,
        )

        for (j in 0 until columns) {
            for (i in 0 until rows) {
                var expected = 0.0
                for (p in 0 until depth) expected += a[i, p] * b[p, j]
                assertEquals(expected, c[i + j * F64PackedPanels.tileRows], 1e-12)
            }
        }
    }

    @Test
    fun `panel sizes reject impossible arrays`() {
        assertFailsWith<IllegalArgumentException> { F64PackedPanels.leftSize(-1, 2) }
        assertFailsWith<IllegalArgumentException> { F64PackedPanels.rightSize(Int.MAX_VALUE, Int.MAX_VALUE) }
    }

    private fun matrix(rows: Int, columns: Int): F64DenseMatrix = F64DenseMatrix.ofColumns(
        Array(columns) { j -> DoubleArray(rows) { i -> 100.0 * j + i + 1.0 } },
    )

    private fun assertPositiveZero(value: Double) {
        assertEquals(0.0.toBits(), value.toBits())
    }
}
