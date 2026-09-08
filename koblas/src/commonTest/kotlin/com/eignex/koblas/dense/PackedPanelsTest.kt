package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import kotlin.math.min
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@OptIn(ExperimentalKoblasApi::class)
class PackedPanelsTest {
    @Test
    fun `left panels round trip across a partial edge`() {
        val tile = PackedPanels.tileRows
        val rows = tile + 1
        val depth = 3
        val source = matrix(rows + 2, depth + 2)
        val packed = DoubleArray(PackedPanels.leftSize(rows, depth) + 4) { Double.NaN }
        val restored = DenseMatrix.zero(rows + 2, depth + 2)

        PackedPanels.packLeft(
            source,
            packed,
            rows,
            depth,
            sourceRow = 1,
            sourceColumn = 1,
            alpha = 2.0,
            destinationOffset = 2,
        )
        PackedPanels.writeLeft(
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
        val columns = PackedPanels.tileColumns + 1
        val source = matrix(columns + 2, depth + 2)
        val packed = DoubleArray(PackedPanels.rightSize(depth, columns))
        val restored = DenseMatrix.zero(columns + 2, depth + 2)

        PackedPanels.packRight(
            source,
            packed,
            depth,
            columns,
            sourceRow = 1,
            sourceColumn = 1,
            transpose = true,
        )
        PackedPanels.writeRight(
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
        val order = min(PackedPanels.tileRows, PackedPanels.tileColumns) + 1
        val source = DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) source[i, j] = if (i >= j) 100.0 * j + i else Double.NaN
        }
        val left = DoubleArray(PackedPanels.leftSize(order, order))
        val right = DoubleArray(PackedPanels.rightSize(order, order))
        val leftRestored = DenseMatrix.zero(order)
        val rightRestored = DenseMatrix.zero(order)

        PackedPanels.packSymmetricLeft(source, left, order, order, lower = true)
        PackedPanels.packSymmetricRight(source, right, order, order, lower = true)
        PackedPanels.writeLeft(left, leftRestored, order, order)
        PackedPanels.writeRight(right, rightRestored, order, order)

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
        val order = min(PackedPanels.tileColumns, 5)
        val source = DenseMatrix.zero(order)
        for (j in 0 until order) {
            for (i in 0 until order) {
                source[i, j] = when {
                    i > j -> 10.0 * j + i
                    else -> Double.NaN
                }
            }
        }
        val packed = DoubleArray(PackedPanels.rightSize(order, order))
        val restored = DenseMatrix.zero(order)

        PackedPanels.packTriangularRight(
            source,
            packed,
            order,
            order,
            lower = true,
            transpose = true,
            unitDiagonal = true,
        )
        PackedPanels.writeRight(packed, restored, order, order)

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
        val rows = PackedPanels.tileRows - 1
        val source = DenseMatrix.zero(rows, 1)
        val packed = DoubleArray(PackedPanels.leftSize(rows, 1))

        PackedPanels.packLeft(source, packed, rows, 1, alpha = Double.POSITIVE_INFINITY)

        for (lane in 0 until rows) assertTrue(packed[lane].isNaN())
        for (lane in rows until PackedPanels.tileRows) assertPositiveZero(packed[lane])
    }

    @Test
    fun `triangular structural zero survives infinite alpha`() {
        val order = min(PackedPanels.tileRows, 3)
        val source = DenseMatrix.zero(order)
        val packed = DoubleArray(PackedPanels.leftSize(order, order))
        val restored = DenseMatrix.zero(order)

        PackedPanels.packTriangularLeft(
            source,
            packed,
            order,
            order,
            lower = true,
            alpha = Double.POSITIVE_INFINITY,
        )
        PackedPanels.writeLeft(packed, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) {
                if (i < j) assertPositiveZero(restored[i, j]) else assertTrue(restored[i, j].isNaN())
            }
        }
    }

    @Test
    fun `same backing packing uses workspace staging`() {
        val order = PackedPanels.tileRows
        val source = matrix(order, order)
        val expected = source.toArray()
        val workspace = Workspace().also { it.reserve(source.data.size, 1) }

        PackedPanels.packLeft(
            source,
            source.data,
            order,
            order,
            transpose = true,
            workspace = workspace,
        )
        val restored = DenseMatrix.zero(order)
        PackedPanels.writeLeft(source.data, restored, order, order)

        for (j in 0 until order) {
            for (i in 0 until order) assertEquals(expected[j][i], restored[i, j])
        }
    }

    @Test
    fun `same backing writeback uses workspace staging`() {
        val order = PackedPanels.tileColumns
        val source = matrix(order, order)
        val packed = DoubleArray(PackedPanels.rightSize(order, order))
        PackedPanels.packRight(source, packed, order, order)
        val destination = DenseMatrix.wrap(order, order, packed.copyOf())
        val workspace = Workspace().also { it.reserve(packed.size, 1) }

        PackedPanels.writeRight(
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
        val rows = PackedPanels.tileRows + 1
        val depth = 2
        val panel = DoubleArray(PackedPanels.leftSize(rows, depth)) { it + 1.0 }
        val before = panel.copyOf()

        PackedPanels.clearLeftPadding(panel, rows, depth)

        val edgePanel = PackedPanels.tileRows * depth
        for (step in 0 until depth) {
            assertEquals(
                before[edgePanel + step * PackedPanels.tileRows],
                panel[edgePanel + step * PackedPanels.tileRows],
            )
            for (lane in 1 until PackedPanels.tileRows) {
                assertPositiveZero(panel[edgePanel + step * PackedPanels.tileRows + lane])
            }
        }
    }

    @Test
    fun `packed panels feed the platform tile`() {
        val rows = PackedPanels.tileRows - 1
        val columns = PackedPanels.tileColumns - 1
        val depth = 3
        val a = matrix(rows, depth)
        val b = matrix(depth, columns)
        val packedA = DoubleArray(PackedPanels.leftSize(rows, depth))
        val packedB = DoubleArray(PackedPanels.rightSize(depth, columns))
        val c = DoubleArray(PackedPanels.tileRows * PackedPanels.tileColumns)

        PackedPanels.packLeft(a, packedA, rows, depth)
        PackedPanels.packRight(b, packedB, depth, columns)
        PlatformKernels.gemmTile(
            depth,
            packedA,
            0,
            packedB,
            0,
            c,
            0,
            PackedPanels.tileRows,
        )

        for (j in 0 until columns) {
            for (i in 0 until rows) {
                var expected = 0.0
                for (p in 0 until depth) expected += a[i, p] * b[p, j]
                assertEquals(expected, c[i + j * PackedPanels.tileRows], 1e-12)
            }
        }
    }

    @Test
    fun `packed trsm solves partial tiles in both directions`() {
        val rows = maxOf(1, PackedPanels.tileRows - 1)
        val order = maxOf(1, PackedPanels.tileColumns - 1)
        for (lower in booleanArrayOf(false, true)) {
            val triangle = triangle(order, lower)
            val expected = matrix(rows, order)
            val rightHandSide = rightProduct(expected, triangle)
            val packedTriangle = DoubleArray(PackedPanels.rightSize(order, order))
            val packedRightHandSide = DoubleArray(PackedPanels.leftSize(rows, order))
            PackedPanels.packTriangularRight(
                triangle, packedTriangle, order, order, lower = lower,
            )
            PackedPanels.packLeft(rightHandSide, packedRightHandSide, rows, order)

            PackedPanels.trsm(packedTriangle, packedRightHandSide, rows, order, lower)

            val actual = DenseMatrix.zero(rows, order)
            PackedPanels.writeLeft(packedRightHandSide, actual, rows, order)
            assertClose(expected, actual, "lower=$lower partial packed solve")
        }
    }

    @Test
    fun `packed gemm trsm accepts exact edge buffers`() {
        val rows = maxOf(1, PackedPanels.tileRows - 1)
        val order = maxOf(1, PackedPanels.tileColumns - 1)
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
        val packedLeft = DoubleArray(PackedPanels.leftSize(rows, depth))
        val packedRight = DoubleArray(PackedPanels.rightSize(depth, order))
        val packedTriangle = DoubleArray(PackedPanels.rightSize(order, order))
        val packedInitial = DoubleArray(PackedPanels.leftSize(rows, order))
        PackedPanels.packLeft(left, packedLeft, rows, depth)
        PackedPanels.packRight(right, packedRight, depth, order)
        PackedPanels.packTriangularRight(triangle, packedTriangle, order, order, lower = true)
        PackedPanels.packLeft(initial, packedInitial, rows, order)

        PackedPanels.gemmTrsm(
            packedLeft, packedRight, packedTriangle, packedInitial, rows, order, depth, lower = true,
        )

        val actual = DenseMatrix.zero(rows, order)
        PackedPanels.writeLeft(packedInitial, actual, rows, order)
        assertClose(expected, actual, "partial fused packed solve", tolerance = 1e-9)
    }

    @Test
    fun `packed trsm stages an aliased triangle`() {
        val rows = min(PackedPanels.tileRows, 2)
        val order = min(PackedPanels.tileColumns, 2)
        val triangle = triangle(order, lower = true)
        val expected = matrix(rows, order)
        val rightHandSide = rightProduct(expected, triangle)
        val triangleSize = PackedPanels.rightSize(order, order)
        val rightHandSideSize = PackedPanels.leftSize(rows, order)
        val shared = DoubleArray(triangleSize + rightHandSideSize)
        PackedPanels.packTriangularRight(triangle, shared, order, order, lower = true)
        PackedPanels.packLeft(
            rightHandSide, shared, rows, order, destinationOffset = triangleSize,
        )
        val workspace = Workspace().also { it.reserve(shared.size, 1) }

        PackedPanels.trsm(
            shared,
            shared,
            rows,
            order,
            lower = true,
            rightHandSideOffset = triangleSize,
            workspace = workspace,
        )

        val actual = DenseMatrix.zero(rows, order)
        PackedPanels.writeLeft(shared, actual, rows, order, sourceOffset = triangleSize)
        assertClose(expected, actual, "aliased packed triangle")
    }

    @Test
    fun `panel sizes reject impossible arrays`() {
        assertFailsWith<IllegalArgumentException> { PackedPanels.leftSize(-1, 2) }
        assertFailsWith<IllegalArgumentException> { PackedPanels.rightSize(Int.MAX_VALUE, Int.MAX_VALUE) }
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
