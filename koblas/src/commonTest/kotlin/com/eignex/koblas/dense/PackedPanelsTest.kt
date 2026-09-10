package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.KoblasContext
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.packedPanels
import com.eignex.koblas.sparse.scalarSparseKernelFamilies
import kotlin.math.min
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class PackedPanelsTest {
    @Test
    fun `explicit engine owns packed panel shape and arithmetic`() {
        var called = false
        val recording = object : PackedKernels by PortablePackedKernels {
            override val gemmTileRows: Int = 2
            override val gemmTileCols: Int = 3

            override fun trsmTile(
                validRows: Int,
                order: Int,
                packedTriangle: DoubleArray,
                triangleOff: Int,
                lower: Boolean,
                unitDiag: Boolean,
                x: DoubleArray,
                xOff: Int,
            ) {
                called = true
                for (column in 0 until order) {
                    for (row in 0 until validRows) x[xOff + row + column * gemmTileRows] /= 2.0
                }
            }
        }
        val engine = KoblasContext(
            DenseKernelFamilies(ScalarKernels, ScalarPanelKernels, recording),
            scalarSparseKernelFamilies,
        )
        val panels = engine.packedPanels
        val triangle = doubleArrayOf(2.0, 0.0, 0.0, 0.0, 2.0, 0.0)
        val rightHandSide = doubleArrayOf(2.0, 4.0, 0.0, 0.0)

        assertEquals(2, panels.tileRows)
        assertEquals(3, panels.tileColumns)
        panels.trsm(triangle, rightHandSide, rows = 2, order = 2, lower = true)

        assertTrue(called)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 0.0, 0.0), rightHandSide)
    }

    @Test
    fun `left panels round trip across a partial edge`() {
        val tile = packedPanels.tileRows
        val rows = tile + 1
        val depth = 3
        val source = matrix(rows + 2, depth + 2)
        val packed = DoubleArray(packedPanels.leftSize(rows, depth) + 4) { Double.NaN }
        val restored = DenseMatrix.zero(rows + 2, depth + 2)

        packedPanels.packLeft(
            source,
            packed,
            rows,
            depth,
            sourceRow = 1,
            sourceColumn = 1,
            alpha = 2.0,
            destinationOffset = 2,
        )
        packedPanels.writeLeft(
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
        val columns = packedPanels.tileColumns + 1
        val source = matrix(columns + 2, depth + 2)
        val packed = DoubleArray(packedPanels.rightSize(depth, columns))
        val restored = DenseMatrix.zero(columns + 2, depth + 2)

        packedPanels.packRight(
            source,
            packed,
            depth,
            columns,
            sourceRow = 1,
            sourceColumn = 1,
            transpose = true,
        )
        packedPanels.writeRight(
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
        val order = min(packedPanels.tileRows, packedPanels.tileColumns) + 1
        val source = DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) source[i, j] = if (i >= j) 100.0 * j + i else Double.NaN
        }
        val left = DoubleArray(packedPanels.leftSize(order, order))
        val right = DoubleArray(packedPanels.rightSize(order, order))
        val leftRestored = DenseMatrix.zero(order)
        val rightRestored = DenseMatrix.zero(order)

        packedPanels.packSymmetricLeft(source, left, order, order, lower = true)
        packedPanels.packSymmetricRight(source, right, order, order, lower = true)
        packedPanels.writeLeft(left, leftRestored, order, order)
        packedPanels.writeRight(right, rightRestored, order, order)

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
        val order = min(packedPanels.tileColumns, 5)
        val source = DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) {
                source[i, j] = when {
                    i > j -> 10.0 * j + i
                    else -> Double.NaN
                }
            }
        }
        val packed = DoubleArray(packedPanels.rightSize(order, order))
        val restored = DenseMatrix.zero(order)

        packedPanels.packTriangularRight(
            source,
            packed,
            order,
            order,
            lower = true,
            transpose = true,
            unitDiagonal = true,
        )
        packedPanels.writeRight(packed, restored, order, order)

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
        val rows = packedPanels.tileRows - 1
        val source = DenseMatrix.zero(rows, 1)
        val packed = DoubleArray(packedPanels.leftSize(rows, 1))

        packedPanels.packLeft(source, packed, rows, 1, alpha = Double.POSITIVE_INFINITY)

        for (lane in 0 until rows) assertTrue(packed[lane].isNaN())
        for (lane in rows until packedPanels.tileRows) assertPositiveZero(packed[lane])
    }

    @Test
    fun `triangular structural zero survives infinite alpha`() {
        val order = min(packedPanels.tileRows, 3)
        val source = DenseMatrix.zero(order)
        val packed = DoubleArray(packedPanels.leftSize(order, order))
        val restored = DenseMatrix.zero(order)

        packedPanels.packTriangularLeft(
            source,
            packed,
            order,
            order,
            lower = true,
            alpha = Double.POSITIVE_INFINITY,
        )
        packedPanels.writeLeft(packed, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) {
                if (i < j) assertPositiveZero(restored[i, j]) else assertTrue(restored[i, j].isNaN())
            }
        }
    }

    @Test
    fun `same backing packing uses workspace staging`() {
        val order = packedPanels.tileRows
        val source = matrix(order, order)
        val expected = source.toArray()
        val workspace = Workspace().also { it.reserve(source.data.size, 1) }

        packedPanels.packLeft(
            source,
            source.data,
            order,
            order,
            transpose = true,
            workspace = workspace,
        )
        val restored = DenseMatrix.zero(order)
        packedPanels.writeLeft(source.data, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) assertEquals(expected[j][i], restored[i, j])
        }
    }

    @Test
    fun `same backing writeback uses workspace staging`() {
        val order = packedPanels.tileColumns
        val source = matrix(order, order)
        val packed = DoubleArray(packedPanels.rightSize(order, order))
        packedPanels.packRight(source, packed, order, order)
        val destination = DenseMatrix.wrap(order, order, packed.copyOf())
        val workspace = Workspace().also { it.reserve(packed.size, 1) }

        packedPanels.writeRight(
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
        val rows = packedPanels.tileRows + 1
        val depth = 2
        val panel = DoubleArray(packedPanels.leftSize(rows, depth)) { it + 1.0 }
        val before = panel.copyOf()

        packedPanels.clearLeftPadding(panel, rows, depth)

        val edgePanel = packedPanels.tileRows * depth
        for (step in 0 until depth) {
            assertEquals(
                before[edgePanel + step * packedPanels.tileRows],
                panel[edgePanel + step * packedPanels.tileRows],
            )
            for (lane in 1 until packedPanels.tileRows) {
                assertPositiveZero(panel[edgePanel + step * packedPanels.tileRows + lane])
            }
        }
    }

    @Test
    fun `packed panels feed the platform tile`() {
        val rows = packedPanels.tileRows - 1
        val columns = packedPanels.tileColumns - 1
        val depth = 3
        val a = matrix(rows, depth)
        val b = matrix(depth, columns)
        val packedA = DoubleArray(packedPanels.leftSize(rows, depth))
        val packedB = DoubleArray(packedPanels.rightSize(depth, columns))
        val c = DoubleArray(packedPanels.tileRows * packedPanels.tileColumns)

        packedPanels.packLeft(a, packedA, rows, depth)
        packedPanels.packRight(b, packedB, depth, columns)
        platformDenseKernelFamilies.packed.gemmTile(
            depth,
            packedA,
            0,
            packedB,
            0,
            c,
            0,
            packedPanels.tileRows,
        )

        for (j in 0 until columns) {
            for (i in 0 until rows) {
                var expected = 0.0
                for (p in 0 until depth) expected += a[i, p] * b[p, j]
                assertEquals(expected, c[i + j * packedPanels.tileRows], 1e-12)
            }
        }
    }

    @Test
    fun `packed trsm solves partial tiles in both directions`() {
        val rows = maxOf(1, packedPanels.tileRows - 1)
        val order = maxOf(1, packedPanels.tileColumns - 1)
        for (lower in booleanArrayOf(false, true)) {
            val triangle = triangle(order, lower)
            val expected = matrix(rows, order)
            val rightHandSide = rightProduct(expected, triangle)
            val packedTriangle = DoubleArray(packedPanels.rightSize(order, order))
            val packedRightHandSide = DoubleArray(packedPanels.leftSize(rows, order))
            packedPanels.packTriangularRight(
                triangle,
                packedTriangle,
                order,
                order,
                lower = lower,
            )
            packedPanels.packLeft(rightHandSide, packedRightHandSide, rows, order)

            packedPanels.trsm(packedTriangle, packedRightHandSide, rows, order, lower)

            val actual = DenseMatrix.zero(rows, order)
            packedPanels.writeLeft(packedRightHandSide, actual, rows, order)
            assertClose(expected, actual, "lower=$lower partial packed solve")
        }
    }

    @Test
    fun `packed gemm trsm accepts exact edge buffers`() {
        val rows = maxOf(1, packedPanels.tileRows - 1)
        val order = maxOf(1, packedPanels.tileColumns - 1)
        val depth = 3
        val triangle = triangle(order, lower = true)
        val expected = matrix(rows, order)
        val left = matrix(rows, depth)
        val right = matrix(depth, order)
        val initial = rightProduct(expected, triangle)
        for (column in 0 until order) {
            for (step in 0 until depth) {
                for (row in 0 until rows) initial[row, column] += left[row, step] * right[step, column]
            }
        }
        val packedLeft = DoubleArray(packedPanels.leftSize(rows, depth))
        val packedRight = DoubleArray(packedPanels.rightSize(depth, order))
        val packedTriangle = DoubleArray(packedPanels.rightSize(order, order))
        val packedInitial = DoubleArray(packedPanels.leftSize(rows, order))
        packedPanels.packLeft(left, packedLeft, rows, depth)
        packedPanels.packRight(right, packedRight, depth, order)
        packedPanels.packTriangularRight(triangle, packedTriangle, order, order, lower = true)
        packedPanels.packLeft(initial, packedInitial, rows, order)

        packedPanels.gemmTrsm(
            packedLeft,
            packedRight,
            packedTriangle,
            packedInitial,
            rows,
            order,
            depth,
            lower = true,
        )

        val actual = DenseMatrix.zero(rows, order)
        packedPanels.writeLeft(packedInitial, actual, rows, order)
        assertClose(expected, actual, "partial fused packed solve", tolerance = 1e-9)
    }

    @Test
    fun `packed trsm stages an aliased triangle`() {
        val rows = min(packedPanels.tileRows, 2)
        val order = min(packedPanels.tileColumns, 2)
        val triangle = triangle(order, lower = true)
        val expected = matrix(rows, order)
        val rightHandSide = rightProduct(expected, triangle)
        val triangleSize = packedPanels.rightSize(order, order)
        val rightHandSideSize = packedPanels.leftSize(rows, order)
        val shared = DoubleArray(triangleSize + rightHandSideSize)
        packedPanels.packTriangularRight(triangle, shared, order, order, lower = true)
        packedPanels.packLeft(
            rightHandSide,
            shared,
            rows,
            order,
            destinationOffset = triangleSize,
        )
        val workspace = Workspace().also { it.reserve(shared.size, 1) }

        packedPanels.trsm(
            shared,
            shared,
            rows,
            order,
            lower = true,
            rightHandSideOffset = triangleSize,
            workspace = workspace,
        )

        val actual = DenseMatrix.zero(rows, order)
        packedPanels.writeLeft(shared, actual, rows, order, sourceOffset = triangleSize)
        assertClose(expected, actual, "aliased packed triangle")
    }

    @Test
    fun `panel sizes reject impossible arrays`() {
        assertFailsWith<IllegalArgumentException> { packedPanels.leftSize(-1, 2) }
        assertFailsWith<IllegalArgumentException> { packedPanels.rightSize(Int.MAX_VALUE, Int.MAX_VALUE) }
    }

    private fun matrix(rows: Int, columns: Int): DenseMatrix = DenseMatrix.ofColumns(
        Array(columns) { j -> DoubleArray(rows) { i -> 100.0 * j + i + 1.0 } },
    )

    private fun triangle(order: Int, lower: Boolean): DenseMatrix = DenseMatrix.zero(order).also { matrix ->
        for (column in 0 until order) {
            for (row in 0 until order) {
                if (if (lower) row >= column else row <= column) {
                    matrix[row, column] = if (row == column) row + 2.0 else 0.1 * (row + column + 1.0)
                } else {
                    matrix[row, column] = Double.NaN
                }
            }
        }
    }

    private fun rightProduct(left: DenseMatrix, triangle: DenseMatrix): DenseMatrix =
        DenseMatrix.zero(left.rows, triangle.cols).also { product ->
            for (column in 0 until triangle.cols) {
                for (inner in 0 until triangle.rows) {
                    val coefficient = triangle[inner, column]
                    if (!coefficient.isNaN()) {
                        for (row in 0 until left.rows) product[row, column] += left[row, inner] * coefficient
                    }
                }
            }
        }

    private fun assertPositiveZero(value: Double) {
        assertEquals(0.0.toBits(), value.toBits())
    }
}
