package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Packed operands: what their layout promises, what they own, and what refuses to read them. The expected
 * physical positions are written out from the contract's index formula rather than taken from the packer.
 */
class PackedMatrixTest {
    private val engines: List<KoblasEngine> get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)

    @Test
    fun `a packed operand reproduces the logical matrix it was packed from`() {
        val rng = Random(20261005)
        for (engine in engines) {
            for (transpose in booleanArrayOf(false, true)) {
                val a = randomMatrix(if (transpose) 7 else 13, if (transpose) 13 else 7, rng)
                val left = engine.packLeft(a, transpose)
                val right = engine.packRight(a, transpose)

                // The logical operand is 13 by 7 either way; what changes is the stride the packer walks.
                assertEquals(13, left.rows, "${engine.name} left rows")
                assertEquals(7, left.columns, "${engine.name} left columns")
                for (i in 0 until left.rows) {
                    for (j in 0 until left.columns) {
                        val expected = if (transpose) a[j, i] else a[i, j]
                        assertClose(expected, left[i, j], "${engine.name} left ($i, $j)")
                        assertClose(expected, right[i, j], "${engine.name} right ($i, $j)")
                    }
                }
            }
        }
    }

    @Test
    fun `a packed operand sits where the index formula says it does`() {
        val rng = Random(20261006)
        val a = randomMatrix(11, 5, rng)
        for (engine in engines) {
            val group = engine.productKernels.tileRows
            val packed = engine.packLeft(a, transpose = false)

            assertEquals(PackedRole.Left, packed.layout.role)
            assertEquals(5 * group, packed.layout.groupStride, "${engine.name} group stride")
            assertEquals((11 + group - 1) / group * 5 * group, packed.layout.storageSize, "${engine.name} size")
            for (i in 0 until 11) {
                for (p in 0 until 5) {
                    val at = i / group * (5 * group) + p * group + i % group
                    assertClose(a[i, p], packed.values[at], "${engine.name} entry ($i, $p)")
                }
            }
        }
    }

    @Test
    fun `the padding a short group leaves is positive zero`() {
        val rng = Random(20261007)
        for (engine in engines) {
            val group = engine.productKernels.tileColumns
            val packed = engine.packRight(randomMatrix(4, group + 1, rng), transpose = false)
            val lastGroup = packed.layout.groupStride

            for (p in 0 until 4) {
                for (lane in 1 until group) {
                    val at = lastGroup + p * group + lane
                    assertEquals(0.0, packed.values[at], "${engine.name} padding at $at")
                    assertTrue(1.0 / packed.values[at] > 0.0, "${engine.name} padding at $at is negative zero")
                }
            }
        }
    }

    @Test
    fun `a packed operand keeps its values when the matrix it came from changes`() {
        val rng = Random(20261008)
        val a = randomMatrix(9, 6, rng)
        val packed = koblas.packLeft(a, transpose = false)
        val before = DoubleArray(9 * 6) { packed[it % 9, it / 9] }

        a.values.fill(Double.NaN)

        for (at in before.indices) {
            assertEquals(before[at], packed[at % 9, at / 9], "a packed operand followed its source")
        }
    }

    @Test
    fun `a layout refuses extents no array can hold`() {
        assertFailsWith<DimensionMismatch> { PackedLayout(PackedRole.Left, 100_000, 100_000, 4) }
    }

    // A left panel of no rows and two billion columns holds nothing, so working out how long a group would
    // have been would overflow an intermediate that nothing reads.
    @Test
    fun `an empty layout has no storage whatever its other extent is`() {
        for (layout in listOf(
            PackedLayout(PackedRole.Left, 0, Int.MAX_VALUE, 4),
            PackedLayout(PackedRole.Right, Int.MAX_VALUE, 0, 4),
            PackedLayout(PackedRole.Left, Int.MAX_VALUE, 0, 4),
            PackedLayout(PackedRole.Right, 0, Int.MAX_VALUE, 4),
        )) {
            assertEquals(0, layout.storageSize, "$layout")
            assertEquals(0, layout.groupStride, "$layout")
        }
    }

    @Test
    fun `a layout refuses a negative extent or a non positive group`() {
        assertFailsWith<IllegalArgumentException> { PackedLayout(PackedRole.Left, -1, 4, 4) }
        assertFailsWith<IllegalArgumentException> { PackedLayout(PackedRole.Right, 4, 4, 0) }
    }

    @Test
    fun `a packed index outside the logical shape is refused`() {
        val packed = koblas.packLeft(randomMatrix(4, 3, Random(20261009)), transpose = false)

        assertFailsWith<IndexOutOfBoundsException> { packed[4, 0] }
        assertFailsWith<IndexOutOfBoundsException> { packed[0, 3] }
        assertFailsWith<IndexOutOfBoundsException> { packed[-1, 0] }
    }

    @Test
    fun `an operand packed for the other side is refused before the destination is touched`() {
        val rng = Random(20261010)
        for (engine in engines) {
            val left = engine.packLeft(randomMatrix(8, 8, rng), transpose = false)
            val right = engine.packRight(randomMatrix(8, 8, rng), transpose = false)
            val c = DenseMatrix(8, 8, DoubleArray(64) { 7.0 })

            assertFailsWith<DimensionMismatch> { engine.gemm(1.0, right, left, 0.0, c) }
            assertTrue(c.values.all { it == 7.0 }, "${engine.name} wrote before refusing a role mismatch")
        }
    }

    @Test
    fun `an operand grouped for another tile is refused before the destination is touched`() {
        val rng = Random(20261011)
        for (engine in engines) {
            val tile = engine.productKernels
            val wrong = packedLeft(randomMatrix(8, 8, rng), transpose = false, group = tile.tileRows + 1)
            val right = engine.packRight(randomMatrix(8, 8, rng), transpose = false)
            val c = DenseMatrix(8, 8, DoubleArray(64) { 7.0 })

            assertFailsWith<DimensionMismatch> { engine.gemm(1.0, wrong, right, 0.0, c) }
            assertTrue(c.values.all { it == 7.0 }, "${engine.name} wrote before refusing a group mismatch")
        }
    }

    @Test
    fun `a packed product refuses shapes that do not conform`() {
        val rng = Random(20261012)
        for (engine in engines) {
            val left = engine.packLeft(randomMatrix(8, 5, rng), transpose = false)
            val right = engine.packRight(randomMatrix(6, 8, rng), transpose = false)
            val c = DenseMatrix(8, 8, DoubleArray(64) { 7.0 })

            assertFailsWith<DimensionMismatch> { engine.gemm(1.0, left, right, 0.0, c) }
            val conforming = engine.packRight(randomMatrix(5, 8, rng), transpose = false)
            assertFailsWith<DimensionMismatch> {
                engine.gemm(1.0, left, conforming, 0.0, DenseMatrix(8, 7, DoubleArray(56)))
            }
            assertTrue(c.values.all { it == 7.0 }, "${engine.name} wrote before refusing a shape mismatch")
        }
    }

    /** Repacking is the caller's explicit step, so an operand packed for one engine is refused by the other. */
    @Test
    fun `a panel packed for one tile is refused by an engine with another`() {
        val simd = BuiltinEngines.simd ?: return skipped()
        val scalar = BuiltinEngines.scalar
        if (simd.productKernels.tileRows == scalar.productKernels.tileRows) return skipped()
        val rng = Random(20261013)
        val left = scalar.packLeft(randomMatrix(16, 16, rng), transpose = false)
        val right = simd.packRight(randomMatrix(16, 16, rng), transpose = false)
        val c = DenseMatrix(16, 16, DoubleArray(256) { 7.0 })

        assertFailsWith<DimensionMismatch> { simd.gemm(1.0, left, right, 0.0, c) }

        val repacked = simd.packLeft(DenseMatrix(16, 16, DoubleArray(256) { left[it % 16, it / 16] }), false)
        simd.gemm(1.0, repacked, right, 0.0, c)
        assertTrue(c.values.none { it == 7.0 }, "the repacked product wrote nothing")
    }

    /**
     * A panel is compatible with the grouping it was packed along and nothing is said about the other: a
     * left panel grouped by eight rows is read by an eight by four tile and by an eight by two tile alike.
     */
    @Test
    fun `a panel is usable by any tile grouped the same way along its own axis`() {
        val rng = Random(20261024)
        val m = 12
        val k = 9
        val n = 7
        for (engine in engines) {
            val narrow = NarrowColumnProducts(engine.productKernels.tileRows)
            val blas = PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels, narrow)
            val a = randomMatrix(m, k, rng)
            val b = randomMatrix(k, n, rng)
            val left = engine.packLeft(a, transpose = false)
            val right = rightPanelAt(b, narrow.tileColumns)
            val expected = DenseMatrix(m, n, DoubleArray(m * n))
            ReferenceBlas.gemm(0.875, a, false, b, false, 0.0, expected)
            val c = DenseMatrix(m, n, DoubleArray(m * n) { Double.NaN })

            blas.gemm(0.875, left, right, 0.0, c)

            assertEquals(engine.productKernels.tileRows, left.layout.group, "${engine.name} left group")
            assertClose(expected.values, c.values, "${engine.name} through ${narrow.name}")
        }
    }

    /** A right panel built by hand at [group], through the layout's own published formula. */
    private fun rightPanelAt(b: DenseMatrix, group: Int): PackedMatrix {
        val layout = PackedLayout(PackedRole.Right, b.rows, b.cols, group)
        val values = DoubleArray(layout.storageSize)
        for (p in 0 until b.rows) for (j in 0 until b.cols) values[layout.index(p, j)] = b[p, j]
        return PackedMatrix(values, layout)
    }

    /**
     * A tile as many rows deep as the engine's and two columns wide, written out from the layout formula so
     * that compatibility can be shown to follow the operand's grouping rather than the whole tile shape.
     */
    private class NarrowColumnProducts(override val tileRows: Int) : DenseProductKernels {
        override val name: String get() = "narrow-tile(${tileRows}x$tileColumns)"

        override val tileColumns: Int get() = 2

        override fun implementationsFor(rows: Int, columns: Int, depth: Int): List<String> =
            if (rows <= 0 || columns <= 0 || depth <= 0) emptyList() else listOf(name)

        override fun packsProduct(rows: Int, columns: Int, depth: Int): Boolean = true

        @Suppress("LongParameterList") // the product block contract
        override fun productBlock(
            alpha: Double,
            packedA: DoubleArray,
            aOffset: Int,
            aGroupStride: Int,
            packedB: DoubleArray,
            bOffset: Int,
            bGroupStride: Int,
            rows: Int,
            columns: Int,
            depth: Int,
            beta: Double,
            c: DoubleArray,
            cOffset: Int,
            ldc: Int,
        ) {
            for (j in 0 until columns) {
                for (i in 0 until rows) {
                    var sum = 0.0
                    for (p in 0 until depth) {
                        val left = packedA[aOffset + i / tileRows * aGroupStride + p * tileRows + i % tileRows]
                        val right =
                            packedB[bOffset + j / tileColumns * bGroupStride + p * tileColumns + j % tileColumns]
                        sum += left * right
                    }
                    val at = cOffset + i + j * ldc
                    c[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * c[at]
                }
            }
        }
    }

    private fun skipped() {
        println("SKIPPED: this runtime has one product tile geometry, so no cross-tile refusal to check")
    }
}
